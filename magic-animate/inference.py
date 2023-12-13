import argparse
import numpy as np
from PIL import Image

from demo.animate import MagicAnimate

def main(reference_image_path, motion_sequence_path, seed, steps, guidance_scale):
    animator = MagicAnimate()
    reference_image = np.array(Image.open(reference_image_path))
    animation = animator(reference_image, motion_sequence_path, seed, steps, guidance_scale)

    return animation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate images using MagicAnimate.")
    parser.add_argument("--image", help="Path to the reference image")
    parser.add_argument("--motion", help="Path to the motion sequence video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--steps", type=int, default=25, help="Sampling steps (default: 25)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (default: 7.5)")

    args = parser.parse_args()

    main(args.image, args.motion, args.seed, args.steps, args.guidance_scale)

    