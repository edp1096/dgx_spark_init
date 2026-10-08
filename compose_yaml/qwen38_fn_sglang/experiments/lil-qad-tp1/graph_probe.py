"""Verify replay consumes changed activations and routing, on a warmed stream."""
import torch


def check_replay(layer, dispatch):
    for name, field in (('w13_weight','w1_fp4'), ('w2_weight','w2_fp4'),
                        ('w13_weight_scale','w1_blockscale'),
                        ('w2_weight_scale','w2_blockscale')):
        actual = getattr(layer,name).untyped_storage().data_ptr()
        canonical = getattr(layer.quant_method._experts,field).untyped_storage().data_ptr()
        assert actual == canonical, f'{name} retains a redundant source allocation'
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer.quant_method.apply(layer, dispatch)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = layer.quant_method.apply(layer, dispatch).hidden_states
    for step in range(3):
        dispatch.hidden_states.mul_(0.8)
        dispatch.topk_output.topk_weights.copy_(
            dispatch.topk_output.topk_weights.flip(-1))
        graph.replay()
        torch.cuda.synchronize()
        expected = layer.quant_method.apply(layer, dispatch).hidden_states
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    print(f'{layer.quant_method.quant_mode}: CUDA Graph changed-input replay passed', flush=True)
