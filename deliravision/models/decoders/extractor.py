from delira import get_backends

if "TORCH" in get_backends():
    import torch

    class ForwardHookTorch:
        def __init__(self, module: torch.nn.Module, overwrite_feats=False):
            """
            Forward hook to extract output from layer

            Parameters
            ----------
            module : torch.nn.Module
                module where output should be extracted from
            overwrite_feats : bool
                whether to overwrite the features if they haven't been
                reset to None, default: True

            """
            self.hook = module.register_forward_hook(self.hook_fn)
            self._overwrite_feats = overwrite_feats
            self._features = None

        def hook_fn(self, module, input, output):
            """

            Parameters
            ----------
            module
            input : torch.Tensor
                input tensor
            output : torch.Tensor
                output tensor

            Returns
            -------
            """
            self.features = output

        def remove(self):
            """
            Remove hook
            """
            self.hook.remove()

        @property
        def features(self):
            """
            Property returning the obtained features and setting the
            resetting the internal attribute to None to correctly
            release the graph.

            Returns
            -------
            :class:`torch.Tensor`
                the features to return

            """
            feats = self._features
            self._features = None
            return feats

        @features.setter
        def features(self, new_feats):
            """
            Sets the new features

            Parameters
            ----------
            new_feats : :class:`torch.Tensor`
                the new features

            Raises
            ------
            AssertionError
                if ``self._features`` is not None and the overwrite attribute
                (``self._overwrite_feats``) is not set.

            """
            assert (self._features is None or self._overwrite_feats), \
                "previous features haven't been resetted"

            self._features = new_feats


    def extract_layers_by_str(model, layers):
        """
        Returns references to layers from model by name

        Parameters
        ----------
        model : torch.nn.Module
            model where layers should be extracted from
        layers : iterable of str
            iterable which contains the names of the respective layers

        Returns
        -------
        list
            list with references to specified layers
        """
        def extract_layer_Torch(model, layer):
            """
            Extract a reference to single layer from model

            Parameters
            ----------
            model : torch.nn.Module
                model where layers should be extracted from
            layer : str
                name of respective layer

            Returns
            -------
            nn.Module
                reference to layer
            """
            submod = model
            if '.' in layer:
                # split actual layer name and 'path' to layer
                prefix, name = layer.rsplit('.', 1)

                # extract layer
                for l in prefix.split('.'):
                    submod = getattr(submod, l)
            else:
                name = layer
            return getattr(submod, name)

        return [extract_layer_Torch(model, l) for l in layers]


    class ExtractorTorch:
        def __init__(self, layers):
            """
            Extract feature maps from backbone network

            Parameters
            ----------
            layers : iterable of nn.Module
                layers where feature maps should be extracted from
            """
            self.hooks = [ForwardHookTorch(l) for l in layers]

        def get_feature_maps(self):
            """
            Get extracted feature maps

            Returns
            -------
            list
                list of feature maps
            """
            return [h.features for h in self.hooks]
