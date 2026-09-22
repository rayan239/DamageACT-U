from .resnet_backbone import make_reference_classifier


def build_post_only(num_classes=4, pretrained=True):
    return make_reference_classifier("post_only", num_classes=num_classes, pretrained=pretrained)
