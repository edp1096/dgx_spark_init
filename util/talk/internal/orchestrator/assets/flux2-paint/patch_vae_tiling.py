"""Force bounded VAE tiles for all image nodes, including LanPaint's direct calls."""
import ast
from pathlib import Path

path = Path('/opt/ComfyUI/comfy/sd.py')
source = path.read_text()
patches = {
    '    def decode(self, samples_in, vae_options={}):\n': '''    def decode(self, samples_in, vae_options={}):
        if self.latent_dim == 2 and samples_in.ndim == 4:
            if vae_options:
                raise ValueError("SparkTalk tiled image VAE does not accept decode options")
            ratio = self.spacial_compression_decode()
            logging.info("SparkTalk VAE tiled decode: 512px, overlap 64px")
            return self.decode_tiled(samples_in, tile_x=512 // ratio, tile_y=512 // ratio, overlap=64 // ratio)
''',
    '    def encode(self, pixel_samples):\n': '''    def encode(self, pixel_samples):
        if self.latent_dim == 2 and pixel_samples.ndim == 4:
            logging.info("SparkTalk VAE tiled encode: 512px, overlap 64px")
            return self.encode_tiled(pixel_samples, tile_x=512, tile_y=512, overlap=64)
''',
    '        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)  # TODO: calculate mem required for tile': '''        tile_shape = list(pixel_samples.shape)
        if dims == 2:
            if tile_y is not None: tile_shape[-2] = min(tile_shape[-2], tile_y)
            if tile_x is not None: tile_shape[-1] = min(tile_shape[-1], tile_x)
        memory_used = self.memory_used_encode(tuple(tile_shape), self.vae_dtype)''',
}
for old, new in patches.items():
    if source.count(old) != 1:
        raise RuntimeError(f'Unexpected ComfyUI source anchor: {old!r}')
    source = source.replace(old, new)
ast.parse(source)
path.write_text(source)
print('Installed image VAE tiling for all encode/decode callers')
