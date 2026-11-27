"""Fast shape and gradient checks; this does not train an experiment."""

from __future__ import annotations

import torch

try:
    from .model_4 import ThreeViewRumorModel
    from .model_5 import JointLoss
except ImportError:
    from model_4 import ThreeViewRumorModel
    from model_5 import JointLoss


def main() -> None:
    torch.manual_seed(42)
    model = ThreeViewRumorModel(
        input_dropout=0.0,
        view_dropout=0.0,
        relation_dropout=0.0,
        fusion_dropout=0.0,
        classifier_dropout=0.0,
    )
    objective = JointLoss(0.20)
    text = torch.randn(4, 1024)
    image = torch.randn(4, 1024)
    caption = torch.randn(4, 1024)
    labels = torch.tensor([0, 1, 0, 1])

    outputs = model(text, image, caption)
    assert outputs["content_stack"].shape == (4, 3, 1024)
    assert outputs["relation_stack"].shape == (4, 3, 1024)
    assert outputs["content_concat"].shape == (4, 3072)
    assert outputs["relation_concat"].shape == (4, 3072)
    assert outputs["content_group"].shape == (4, 1024)
    assert outputs["relation_group"].shape == (4, 1024)
    assert outputs["z_content"].shape == (4, 256)
    assert outputs["z_relation"].shape == (4, 256)
    assert outputs["content_relation_difference"].shape == (4, 256)
    assert outputs["content_relation_product"].shape == (4, 256)
    assert outputs["fused"].shape == (4, 1024)
    assert outputs["logits"].shape == (4, 2)
    reconstructed = torch.cat(
        (
            outputs["z_content"],
            outputs["z_relation"],
            outputs["content_relation_difference"],
            outputs["content_relation_product"],
        ),
        dim=-1,
    )
    assert torch.equal(outputs["fused"], reconstructed)
    assert torch.isfinite(outputs["fused"]).all()

    loss, _ = objective(outputs, labels)
    assert torch.isfinite(loss)
    loss.backward()
    content_grad = model.fusion.content_projection[1].weight.grad
    relation_grad = model.fusion.relation_projection[1].weight.grad
    content_group_grad = model.fusion.content_group_encoder.encoder[1].weight.grad
    relation_group_grad = model.fusion.relation_group_encoder.encoder[1].weight.grad
    assert content_grad is not None and torch.isfinite(content_grad).all()
    assert relation_grad is not None and torch.isfinite(relation_grad).all()
    assert content_group_grad is not None and torch.isfinite(content_group_grad).all()
    assert relation_group_grad is not None and torch.isfinite(relation_group_grad).all()
    assert torch.count_nonzero(content_grad).item() > 0
    assert torch.count_nonzero(relation_grad).item() > 0
    assert torch.count_nonzero(content_group_grad).item() > 0
    assert torch.count_nonzero(relation_group_grad).item() > 0

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"smoke_test=passed parameters={parameter_count}")


if __name__ == "__main__":
    main()
