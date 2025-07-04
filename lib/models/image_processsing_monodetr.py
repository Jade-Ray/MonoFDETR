"""Image processor class for MonoDETR."""
from typing import List, Union, Optional, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import torch

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import (
    rescale,
    resize,
    normalize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    AnnotationType,
    ImageInput, 
    PILImageResampling,
    get_image_size,
    ChannelDimension, 
    is_scaled_image, 
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import logging, TensorType

from utils.box_ops import box_cxcylrtb_to_xyxy, box_xyxy_to_cxcywh


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (height <= width and height == size) or (width <= height and width == size):
        return height, width

    if width < height:
        ow = size
        oh = int(size * height / width)
    else:
        oh = size
        ow = int(size * width / height)
    return (oh, ow)


def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int]],
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size. If the desired output size
    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output
    image size is computed by keeping the aspect ratio of the input image size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]` or `List[int]`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.
    """
    image_size = get_image_size(input_image, input_data_format)
    if isinstance(size, (list, tuple)):
        return size

    return get_size_with_aspect_ratio(image_size, size, max_size)


def resize_annotation(
    annotation: Dict[str, Any],
    orig_size: Tuple[int, int],
    target_size: Tuple[int, int],
    threshold: float = 0.5,
    resample: PILImageResampling = PILImageResampling.NEAREST, # type: ignore
):
    """
    Resizes an annotation to a target size.

    Args:
        annotation (`Dict[str, Any]`):
            The annotation dictionary.
        orig_size (`Tuple[int, int]`):
            The original size of the input image.
        target_size (`Tuple[int, int]`):
            The target size of the image, as returned by the preprocessing `resize` step.
        threshold (`float`, *optional*, defaults to 0.5):
            The threshold used to binarize the segmentation masks.
        resample (`PILImageResampling`, defaults to `PILImageResampling.NEAREST`):
            The resampling filter to use when resizing the masks.
    """
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    new_annotation = annotation.copy()
    
    new_annotation["targets"]["boxes"] = annotation["targets"]["boxes"] * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)   # (cx, cy, w, h)
    new_annotation["targets"]["boxes_3d"] = annotation["targets"]["boxes_3d"] * np.asarray([ratio_width, ratio_height, ratio_width, ratio_width, ratio_height, ratio_height], dtype=np.float32)   # (cx, cy, l, r, t, b)
    
    new_annotation["info"]["size"] = target_size
    new_annotation["info"]["corners_3d"] = annotation["info"]["corners_3d"] * np.asarray([ratio_width, ratio_height], dtype=np.float32)   # (N, 8, 2)

    return new_annotation


class MonoDETRImageProcessor(BaseImageProcessor):
    
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR, # type: ignore
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        **kwargs,
    ) -> None:
        size = size if size is not None else {'width': 1280, 'height': 384}
        size = get_size_dict(size, default_to_square=False)
        
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR, # type: ignore
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary containing the size to resize to. Can contain the keys `shortest_edge` and `longest_edge` or
                `height` and `width`.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use if resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        return image

    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PILImageResampling = PILImageResampling.NEAREST, # type: ignore
    ) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)
    
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Rescale the image by the given factor. image = image * rescale_factor.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            rescale_factor (`float`):
                The value to use for rescaling.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image. Can be
                one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize 2d boxes and 3d boxes to the range [0, 1].
        """
        H, W = image_size
        norm_annotation = annotation.copy()
        norm_annotation["targets"]["boxes"] = annotation["targets"]["boxes"] / np.array([W, H, W, H], dtype=np.float32)   # (cx, cy, w, h)
        norm_annotation["targets"]["boxes_3d"] = annotation["targets"]["boxes_3d"] / np.array([W, H, W, W, H, H], dtype=np.float32)   # (cx, cy, l, r, t, b)
        return norm_annotation

    def preprocess(
        self, 
        images: ImageInput, 
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,  # PILImageResampling
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> BatchFeature:
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size=size, default_to_square=False)
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        
        if do_resize is not None and size is None:
            raise ValueError("Size and max_size must be specified if do_resize is True.")
        
        if do_rescale is not None and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")
        
        if do_normalize is not None and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        images = make_list_of_images(images)
        if annotations is not None and isinstance(annotations, dict):
            annotations = [annotations]
        
        if annotations is not None and len(images) != len(annotations):
            raise ValueError(
                f"The number of images ({len(images)}) and annotations ({len(annotations)}) do not match."
            )
        
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
            
        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])
        
        # transformations
        if do_resize:
            if annotations is not None:
                resized_images, resized_annotations = [], []
                for image, target in zip(images, annotations):
                    orig_size = get_image_size(image, input_data_format)
                    resized_image = self.resize(
                        image, size=size, resample=resample, input_data_format=input_data_format
                    )
                    resized_annotation = self.resize_annotation(
                        target, orig_size, get_image_size(resized_image, input_data_format)
                    )
                    resized_images.append(resized_image)
                    resized_annotations.append(resized_annotation)
                images = resized_images
                annotations = resized_annotations
                del resized_images, resized_annotations
            else:
                images = [
                    self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
                    for image in images
                ]
        
        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]

        if do_normalize:
            images = [
                self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images
            ]
            if annotations is not None:
                annotations = [
                    self.normalize_annotation(annotation, get_image_size(image, input_data_format))
                    for annotation, image in zip(annotations, images)
                ]
        
        images = [
                to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
                for image in images
            ]
        data = {"pixel_values": images}
        
        if annotations is not None:
            data.update({"calibs": [annotation["calib"] for annotation in annotations]})
        
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        if annotations is not None:
            encoded_inputs["captions"] = [a["caption"] for a in annotations]
            encoded_inputs["targets"] = [
                BatchFeature(annotation["targets"], tensor_type=return_tensors) for annotation in annotations
            ]
            encoded_inputs["info"] = [
                BatchFeature(annotation["info"], tensor_type=return_tensors) for annotation in annotations
            ]
        
        return encoded_inputs

    def post_process_3d_object_detection(
        self, outputs, calibs: TensorType, threshold: float = 0.0, target_sizes: Union[TensorType, List[Tuple]] = None, img_ids: List[int] = None, top_k: int = 50
    ):
        """
        Convert the raw output of [`MonoDETRForMultiObjectDetectionOutput`] into final 3D bounding boxes in (center_x, center_y, center_z, height, width, length) format.
        
        Args:
            outputs (`[MonoDETRForMultiObjectDetectionOutput]`):
                Raw outputs of the model.
            calibs (`TensorType`):
                The P2 calibration matrix of shape `(batch_size, 3, 4)`.
            threshold (`float`, *optional*, defaults to 0.0):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (width, height) of each image in the batch. If left to None, predictions will not be resized.
            img_ids (`List[int]`, *optional*):
                List of image ids.
            top_k (`int`, *optional*, defaults to 50):
                Keep only top k bounding boxes before filtering by thresholding.
        
        Returns:
            `Dict[List]`: A dictionary of list, each list containing the scores, labels, 2Dboxes, 3Dboxes and y-rotations for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes
        
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
        
        prob = out_logits.sigmoid()
        prob = prob.view(out_logits.shape[0], -1)
        k_value = min(top_k, prob.size(1))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor").unsqueeze(-1)
        labels = topk_indexes % out_logits.shape[2]
        
        out_angle, out_size_3d, out_depth = outputs.pred_angle, outputs.pred_3d_dim, outputs.pred_depth
        
        boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6)) # b, tk, 6
        angle_heading = torch.gather(out_angle, 1, topk_boxes.repeat(1, 1, 24)) # b, tk, 24 
        depth = torch.gather(out_depth, 1, topk_boxes.repeat(1, 1, 2)) # b, tk, 2
        size_3d = torch.gather(out_size_3d, 1, topk_boxes.repeat(1, 1, 3)) # b, tk, 3 => h, w, l
        # and from relative [0, 1] to absolute [0, height] coordinates
        if isinstance(target_sizes, List):
            img_w = torch.Tensor([i[0] for i in target_sizes])
            img_h = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_w, img_h = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device) # b, 4
        
        # 2d bboxes decoding
        boxes_2d = box_cxcylrtb_to_xyxy(boxes) * scale_fct[:, None, :] # b, tk, 4
        cx_2d = box_xyxy_to_cxcywh(boxes_2d)[..., 0] # b, tk
        
        # 3d bboxes (dimensions + positions) decoding
        cx_3d = boxes[..., 0] * img_w[:, None] # b, tk
        cy_3d = boxes[..., 1] * img_h[:, None] # b, tk
        depth_rect = depth[..., 0] # b, tk
        depth_sigma = torch.exp(-depth[..., 1]) # b, tk
        
        cu, fu = calibs[:, 0, 2][:, None], calibs[:, 0, 0][:, None]
        tx = calibs[:, 0, 3][:, None] / -fu
        cv, fv = calibs[:, 1, 2][:, None], calibs[:, 1, 1][:, None]
        ty = calibs[:, 1, 3][:, None] / -fv
        rect_x = (cx_3d - cu) * depth_rect / fu + tx
        rect_y = (cy_3d - cv) * depth_rect / fv + ty
        rect_z = depth_rect
        pos_rect = torch.stack([rect_x, rect_y, rect_z], dim=2) # b, tk, 3
        pos_rect[..., 1] += size_3d[..., 0] / 2 # (x, y, z) -> (x, y + h/2, z)
        boxes_3d = torch.cat([size_3d, pos_rect], dim=2) # b, tk, 6 => h, w, l, x, y, z
        # angle heading decoding
        heading_bin, heading_res = angle_heading.split([12, 12], dim=2)
        cls_ind = torch.argmax(heading_bin, dim=2) # b, tk
        heading_res = torch.gather(heading_res, 2, cls_ind.unsqueeze(2)).squeeze(2) # b, tk
        angle = (cls_ind.float() * (2 * torch.pi / 12) + heading_res) # b, tk
        angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)

        rotation_y = angle + torch.arctan2(cx_2d - cu, fu)
        rotation_y = torch.where(rotation_y > torch.pi, rotation_y - 2 * torch.pi, rotation_y)
        rotation_y = torch.where(rotation_y < -torch.pi, rotation_y + 2 * torch.pi, rotation_y)
        
        results = {}
        for i, id in enumerate(img_ids): # batch
            preds = []
            for j, s in enumerate(scores[i]): # max_dets
                if s < threshold:
                    continue
                preds.append({
                    "scores": s * depth_sigma[i, j],
                    "labels": labels[i, j],
                    "boxes_2d": boxes_2d[i, j],
                    "boxes_3d": boxes_3d[i, j],
                    "angle": angle[i, j],
                    "rotation_y": rotation_y[i, j],
                    "query_index": topk_boxes[i, j, 0].item(),  # query index
                })
            results[id] = preds
        return results
    
    def post_process_targets(self, targets: List[Dict], calibs: TensorType, target_sizes: Union[TensorType, List[Tuple]] = None,):
        if target_sizes is not None:
            if len(targets) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the targets"
                )
        
        results = []
        for target, calib, size in zip(targets, calibs, target_sizes):
            if len(target['labels']) == 0:
                results.append(target)
            else:
                scale_fct = torch.cat([size, size], dim=0) # (width, height, width, height)
                size_3d = target['size_3d'] # (N, 3) => (h, w, l)
                boxes = target['boxes_3d'] # (N, 6) => (cx, cy, l, r, t, b)
                boxes_2d = box_cxcylrtb_to_xyxy(target['boxes_3d']) * scale_fct[None, :]  # (N, 4)
                cx_2d = box_xyxy_to_cxcywh(boxes_2d)[..., 0]  # (N,)
                cx_3d = (boxes[..., 0] * size[0])[:, None]  # (N, 1)
                cy_3d = (boxes[..., 1] * size[1])[:, None]  # (N, 1)
                depth_rect = target['depth'] # (N, 1)
                
                cu, fu = calib[0, 2], calib[0, 0]  # (1,)
                tx = calib[0, 3] / -fu
                cv, fv = calib[1, 2], calib[1, 1]
                ty = calib[1, 3] / -fv
                rect_x = (cx_3d - cu) * depth_rect / fu + tx
                rect_y = (cy_3d - cv) * depth_rect / fv + ty
                rect_z = depth_rect
                pos_rect = torch.cat([rect_x, rect_y, rect_z], dim=1) # (N, 3)
                pos_rect[:, 1] += size_3d[:, 0] / 2  # (x, y, z) -> (x, y + h/2, z)
                boxes_3d = torch.cat([size_3d, pos_rect], dim=1)  # (N, 6) => (h, w, l, x, y, z)
                
                angle = target['heading_bin'] * (2 * torch.pi / 12) + target['heading_res']  # (N,)
                angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle).squeeze()
                rotation_y = angle + torch.arctan2(cx_2d - cu, fu)
                rotation_y = torch.where(rotation_y > torch.pi, rotation_y - 2 * torch.pi, rotation_y)
                rotation_y = torch.where(rotation_y < -torch.pi, rotation_y + 2 * torch.pi, rotation_y)

                results.append({
                    "labels": target['labels'],
                    "boxes_2d": boxes_2d,
                    "boxes_3d": boxes_3d,
                    "angle": angle,
                    "rotation_y": rotation_y,
                    "obj_region": target['obj_region'],
                })
        return results
    
    @classmethod
    def save_results(cls, results: Dict[int, List[Dict]], output_path: Path, id2label: Dict) -> None:
        """
        Save the results to a file.
        """
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        for img_id in results.keys():
            with open(output_path / f"{img_id:06d}.txt", "w") as f:
                for pred in results[img_id]:
                    f.write(f'{id2label[pred["labels"].item()]} 0.0 0')
                    f.write(f' {pred["angle"]:.2f}')
                    for box in pred["boxes_2d"]:
                        f.write(f' {box:.2f}')
                    for box in pred["boxes_3d"]:
                        f.write(f' {box:.2f}')
                    f.write(f' {pred["rotation_y"]:.2f}')
                    f.write(f' {pred["scores"]:.2f}')
                    f.write('\n')

    @classmethod
    def prepare_batch(cls, batch: Dict[str, Any], return_label: bool = True, return_info: bool = False, return_pixel_values_without_pd: bool = False) -> Dict[str, Any]:
        """
        Prepare the batch.
        """
        labels_list = []
        mask = batch["targets"]["mask_2d"]
        key_list = ['labels', 'boxes', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(mask.shape[0]):
            labels_dict = {}
            for key, val in batch["targets"].items():
                if key in key_list:
                    labels_dict[key] = val[bz][mask[bz]]
                if key == 'obj_region':
                    labels_dict[key] = val[bz]
            labels_list.append(labels_dict)

        output = {
            "pixel_values": batch["pixel_values"],
            "calibs": batch["calibs"],
            "img_sizes": batch["info"]["img_size"],
        }
        
        if return_label:
            output.update({"labels": labels_list})
        
        if return_info:
            output.update({"info": batch["info"]})
            
        if return_pixel_values_without_pd:
            output.update({
                "pixel_values_without_pd": batch["pixel_values_without_pd"],
                "pixel_values_random_mix": batch["pixel_values_random_mix"],
            })
        
        return output


def dictlist_to_listdict(dictlist: Dict[str, List], batch_size: int = None) -> List[Dict[str, List]]:
    """Convert a dictionary of lists to a list of dictionaries."""
    batch_size = batch_size if batch_size is not None else len(list(dictlist.values())[0])
    return [{key: value[i] for key, value in dictlist.items()} for i in range(batch_size)]
