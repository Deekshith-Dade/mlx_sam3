import mlx.nn as nn

from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model.model_misc import MLP, DotProductScoring

class Sam3Image(nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2
    
    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder=None,
        segmentation_head=None,
        num_feature_levels=1,
        dot_prod_scoring=None,

        use_dot_prod_scoring=True,
        separate_scorer_for_instance: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = self.segmentation_head

        # verify the number of queries for O2O and O2M
        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

        
        self.dot_prod_scoring = dot_prod_scoring
        self.use_dot_prod_scoring = use_dot_prod_scoring

        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.dot_prod_scoring = dot_prod_scoring
            self.instance_dot_prod_scoring = None
            if separate_scorer_for_instance:
                d_prompt_mlp = self.dot_prod_scoring.prompt_mlp
                prompt_mlp = MLP(
                    input_dim=d_prompt_mlp.input_dim,
                    hidden_dim=d_prompt_mlp.hidden_dim,
                    output_dim=d_prompt_mlp.output_dim,
                    num_layers=d_prompt_mlp.num_layers,
                    dropout=d_prompt_mlp.dropout,
                    residual=d_prompt_mlp.residual,
                    out_norm=nn.LayerNorm(256)
                )
                self.instance_dot_prod_scoring = DotProductScoring(
                    d_model=self.dot_prod_scoring.d_model,
                    d_proj=self.dot_prod_scoring.d_proj,
                    prompt_mlp=prompt_mlp
                )
        else:
            self.class_embed = nn.Linear(self.hidden_dim, 1)
            self.instance_class_embed = None
            if separate_scorer_for_instance:
                self.instance_class_embed = nn.Linear(self.hidden_dim, 1)
        
    
    def __call__(self):
        pass