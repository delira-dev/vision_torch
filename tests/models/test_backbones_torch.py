import unittest
import gc


class TestBackbones(unittest.TestCase):

    def setUp(self) -> None:
        test_cases = []

        # TODO: Add MobileNet to tests
        from deliravision.torch.models.backbones import SqueezeNetTorch, AlexNetTorch
        from deliravision.torch.models.model_fns import create_vgg_torch, create_resnet_torch, create_densenet_torch, \
            create_resnext_torch, create_seresnext_torch, create_seresnet_torch

        test_cases.append({
            "network_cls": SqueezeNetTorch,
            "network_kwargs": {'version': 1.0, "num_classes": 1000,
                               "in_channels": 3, "n_dim": 2,
                               "pool_type": "Max", "p_dropout": 0.5},
            "input_shape": (5, 3, 224, 224),
            "name": "SqueezeNet1.0"
        })
        test_cases.append({
            "network_cls": SqueezeNetTorch,
            "network_kwargs": {'version': 1.1, "num_classes": 1000,
                               "in_channels": 3, "n_dim": 2,
                               "pool_type": "Max", "p_dropout": 0.5},
            "input_shape": (5, 3, 224, 224),
            "name": "SqueezeNet1.1"
        })
        test_cases.append({
            "network_cls": AlexNetTorch,
            "network_kwargs": {"num_classes": 1000, "in_channels": 3,
                               "n_dim": 2, "pool_type": "Max"},
            "input_shape": (5, 3, 224, 224),
            "name": "AlexNet"
        })
        test_cases.append({
            "network_cls": create_vgg_torch,
            "network_kwargs": {"num_layers": 11},
            "input_shape": (5, 3, 224, 224),
            "name": "VGG11"
        })
        test_cases.append({
            "network_cls": create_vgg_torch,
            "network_kwargs": {"num_layers": 13},
            "input_shape": (5, 3, 224, 224),
            "name": "VGG13"
        })
        test_cases.append({
            "network_cls": create_vgg_torch,
            "network_kwargs": {"num_layers": 16},
            "input_shape": (5, 3, 224, 224),
            "name": "VGG16"
        })
        test_cases.append({
            "network_cls": create_vgg_torch,
            "network_kwargs": {"num_layers": 19},
            "input_shape": (5, 3, 224, 224),
            "name": "VGG19"
        })
        test_cases.append({
            "network_cls": create_resnet_torch,
            "network_kwargs": {"num_layers": 18,
                               "zero_init_residual": True},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNet18"
        })
        test_cases.append({
            "network_cls": create_resnet_torch,
            "network_kwargs": {"num_layers": 34,
                               "zero_init_residual": True},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNet34"
        })
        test_cases.append({
            "network_cls": create_resnet_torch,
            "network_kwargs": {"num_layers": 26,
                               "zero_init_residual": True},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNet26"
        })
        test_cases.append({
            "network_cls": create_resnet_torch,
            "network_kwargs": {"num_layers": 50,
                               "zero_init_residual": True},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNet50"
        })
        test_cases.append({
            "network_cls": create_resnet_torch,
            "network_kwargs": {"num_layers": 101,
                               "zero_init_residual": True},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNet101"
        })
        test_cases.append({
            "network_cls": create_resnet_torch,
            "network_kwargs": {"num_layers": 152,
                               "zero_init_residual": True},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNet152"
        })
        test_cases.append({
            "network_cls": create_densenet_torch,
            "network_kwargs": {"num_layers": 121,
                               "drop_rate": 0.2},
            "input_shape": (5, 3, 224, 224),
            "name": "DenseNet121"
        })
        test_cases.append({
            "network_cls": create_densenet_torch,
            "network_kwargs": {"num_layers": 161,
                               "drop_rate": 0.2},
            "input_shape": (5, 3, 224, 224),
            "name": "DenseNet161"
        })
        test_cases.append({
            "network_cls": create_densenet_torch,
            "network_kwargs": {"num_layers": 169,
                               "drop_rate": 0.2},
            "input_shape": (5, 3, 224, 224),
            "name": "DenseNet169"
        })
        test_cases.append({
            "network_cls": create_densenet_torch,
            "network_kwargs": {"num_layers": 201,
                               "drop_rate": 0.2},
            "input_shape": (5, 3, 224, 224),
            "name": "DenseNet201"
        })
        test_cases.append({
            "network_cls": create_resnext_torch,
            "network_kwargs": {"num_layers": 26,
                               "start_mode": "7x7"},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNeXt26"
        })
        test_cases.append({
            "network_cls": create_resnext_torch,
            "network_kwargs": {"num_layers": 50,
                               "start_mode": "3x3"},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNeXt50"
        })
        test_cases.append({
            "network_cls": create_resnext_torch,
            "network_kwargs": {"num_layers": 101,
                               "start_mode": "7x7"},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNeXt101"
        })
        test_cases.append({
            "network_cls": create_resnext_torch,
            "network_kwargs": {"num_layers": 152,
                               "start_mode": "3x3"},
            "input_shape": (5, 3, 224, 224),
            "name": "ResNeXt152"
        })
        test_cases.append({
            "network_cls": create_seresnet_torch,
            "network_kwargs": {"num_layers": 18},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNet18"
        })
        test_cases.append({
            "network_cls": create_seresnet_torch,
            "network_kwargs": {"num_layers": 34},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNet34"
        })
        test_cases.append({
            "network_cls": create_seresnet_torch,
            "network_kwargs": {"num_layers": 26},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNet26"
        })
        test_cases.append({
            "network_cls": create_seresnet_torch,
            "network_kwargs": {"num_layers": 50},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNet50"
        })
        test_cases.append({
            "network_cls": create_seresnet_torch,
            "network_kwargs": {"num_layers": 101},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNet101"
        })
        test_cases.append({
            "network_cls": create_seresnet_torch,
            "network_kwargs": {"num_layers": 152},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNet152"
        })

        test_cases.append({
            "network_cls": create_seresnext_torch,
            "network_kwargs": {"num_layers": 26,
                               "cardinality": 1},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNeXt26"
        })
        test_cases.append({
            "network_cls": create_seresnext_torch,
            "network_kwargs": {"num_layers": 50},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNeXt50"
        })
        test_cases.append({
            "network_cls": create_seresnext_torch,
            "network_kwargs": {"num_layers": 101},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNeXt101"
        })
        test_cases.append({
            "network_cls": create_seresnext_torch,
            "network_kwargs": {"num_layers": 152},
            "input_shape": (5, 3, 224, 224),
            "name": "SEResNeXt152"
        })

        self.test_cases = test_cases

    def test_models_forward(self):
        import torch
        device = torch.device("cpu")
        gpu_available = False

        # if torch.cuda.is_available():
        #     gpu_available = True
        #     device = torch.device("cuda:0")

        print("Testing Model Inference:")
        for case in self.test_cases:
            with self.subTest(case=case):
                with torch.no_grad():

                    model = case["network_cls"](**case["network_kwargs"]
                                                ).to(device)
                    input_tensor = torch.rand(case["input_shape"]
                                              ).to(device)

                    result = model(input_tensor)

                    self.assertIsInstance(result, dict)

                    if gpu_available:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    del model
                    del input_tensor
                    del result
                    gc.collect()
                    print("\t%s" % case["name"])


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    unittest.main()
