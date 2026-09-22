from .resnet_backbone import make_reference_classifier


def build_siamese(num_classes=4, pretrained=True):
    return make_reference_classifier("siamese", num_classes=num_classes, pretrained=pretrained)
