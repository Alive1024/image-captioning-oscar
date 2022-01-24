import os
import sys
import os.path as op
sys.path.insert(0, op.dirname(op.dirname(op.abspath(__file__))))

import argparse
import numpy as np
import cv2
import torch
from pytorch_transformers import BertConfig, BertTokenizer
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference_single_image

from oscar.datasets.caption_tsv import CaptionTensorizer
from oscar.modeling.modeling_bert import BertForImageCaptioning
from oscar.utils.misc import set_seed, parse_args

# Root of detectron2:
D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__))
# Settings for the number of features per image. 
# To re-create pretrained features with 36 features per image
NUM_OBJECTS = 36


def restore_training_settings_while_testing(args):
    checkpoint = args.eval_model_dir

    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        print('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
                max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
            'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                print('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def read_visual_genome_categories(objects_vocab_path):
    vg_classes = []
    with open(objects_vocab_path) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())
    return np.array(vg_classes)


def build_bottom_up_attention_model(model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(op.join(D2_ROOT, "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml"))
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # cfg.INPUT.MIN_SIZE_TEST = 600
    # cfg.INPUT.MAX_SIZE_TEST = 1000
    # cfg.MODEL.RPN.NMS_THRESH = 0.7
    # Find a model from detectron2's model zoo
    # cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    cfg.MODEL.WEIGHTS = model_weight_path

    predictor = DefaultPredictor(cfg)
    return predictor


def extract_image_features_od_labels(img_path, predictor, vg_classes):
    raw_image = cv2.imread(img_path)

    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break
        
        # Instances(num_instances=36, image_height=480, image_width=640, fields=[pred_boxes, scores, pred_classes])
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()

        instances = instances.to('cpu')
        roi_features = roi_features.to('cpu')
        boxes = instances.pred_boxes.tensor.numpy()

        # Bottom-Up Attention generates features of size 2048, but Oscar expects 2054.
        # The difference comes from concatenating information about the bounding boxes' positions
        # (Ref: last paragraph on pg. 4 of the original research paper: https://arxiv.org/pdf/2004.06165.pdf).
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / raw_width
        scaled_height = box_height / raw_height
        scaled_x = boxes[:, 0] / raw_width
        scaled_y = boxes[:, 1] / raw_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate(
            (scaled_x,
                scaled_y,
                scaled_x + scaled_width,
                scaled_y + scaled_height,
                scaled_width,
                scaled_height),
            axis=1)
        full_features = np.concatenate((roi_features, spatial_features), axis=1)

        od_labels = ' '.join(vg_classes[instances.pred_classes])
        
        return full_features, od_labels


def infer(img_path, predictor, vg_classes,
          args, model, tokenizer, tensorizer, inputs_param):

    features, od_labels = extract_image_features_od_labels(img_path=img_path,
                                                           predictor=predictor,
                                                           vg_classes=vg_classes)

    with torch.no_grad():
        tensorized_example = tensorizer.tensorize_example(text_a='',
                                                          img_feat=torch.tensor(features),
                                                          text_b=od_labels)

        batch = [torch.unsqueeze(i, dim=0).to(args.device) for i in tensorized_example]
        inputs = {
            'input_ids': batch[0], 'attention_mask': batch[1],
            'token_type_ids': batch[2], 'img_feats': batch[3],
            'masked_pos': batch[4]
        }

        if args.use_cbs:
            inputs.update({
                'fsm': batch[5],
                'num_constraints': batch[6],
            })
        inputs.update(inputs_param)
        outputs = model(**inputs)
        all_caps = outputs[0]  # batch_size * num_keep_best * max_len
        all_confs = torch.exp(outputs[1])

        predictions = []
        for caps, confs in zip(all_caps, all_confs):
            for cap, conf in zip(caps, confs):
                cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
            predictions.append(f'caption: {cap}, conf: {conf.item()}')

        return predictions


def prepare(run_from_cmd=False, eval_model_dir='',
     bottom_up_attention_model_weights='inference_models/faster_rcnn_from_caffe.pkl', vg_objects_vocab='objects_vocab.txt'):

    if (not op.isdir(eval_model_dir)) or eval_model_dir == '':
        raise ValueError('Improper argument: `eval_model_dir`')
    if not op.exists(bottom_up_attention_model_weights):
        raise ValueError('Weights for Bottom-Up_Attention NOT FOUND.')
    if not op.exists(vg_objects_vocab):
        raise ValueError('Object vocab(objects_vocab.txt) for Visual Genome dataset NOT FOUND.')

    # Process intial arguments
    parser = argparse.ArgumentParser()
    if run_from_cmd:
        parser.add_argument("--image_path", type=str, help="The file path of the target image.")
    args = parse_args(parser)
    args.do_test = True
    args.add_od_labels = True
    args.num_beams = 5
    args.per_gpu_eval_batch_size = 1
    if not run_from_cmd:
        args.eval_model_dir = eval_model_dir

    set_seed(88, 1)
    args = restore_training_settings_while_testing(args)

    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    checkpoint = args.eval_model_dir
    config = config_class.from_pretrained(checkpoint)
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    model = model_class.from_pretrained(checkpoint, config=config)

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(args.device)
    model.eval()

    tensorizer = CaptionTensorizer(tokenizer, max_img_seq_length=args.max_img_seq_length,
                max_seq_length=args.max_seq_length, max_seq_a_length=args.max_gen_length,
                mask_prob=0.15, max_masked_tokens=3, is_train=False)
    predictor = build_bottom_up_attention_model(bottom_up_attention_model_weights)
    vg_classes = read_visual_genome_categories(vg_objects_vocab)

    # Constrcut inputs_param
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token, 
                                        tokenizer.pad_token, tokenizer.mask_token, '.'])
    inputs_param = {
        'is_decode': True,
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels,
        'od_labels_start_posid': args.max_seq_a_length,

        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.num_beams,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_keep_best": args.num_keep_best,
    }
    if args.use_cbs:
        inputs_param.update({'use_cbs': True,
            'min_constraints_to_satisfy': args.min_constraints_to_satisfy,
        })

    return predictor, vg_classes, args, model, tokenizer, tensorizer, inputs_param


def main():
    # --image_path & --eval_model_dir are necessary when running from cmd
    predictor, vg_classes, args, model, tokenizer, tensorizer, inputs_param = prepare(run_from_cmd=True)
    res = infer(args.image_path, predictor, vg_classes,
          args, model, tokenizer, tensorizer, inputs_param)
    print(res)


if __name__ == '__main__':
    main()
