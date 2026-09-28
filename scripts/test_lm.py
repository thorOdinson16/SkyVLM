import os
import sys
import time

import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from lm import (
    SkyVLMForCausalLM,
    count_parameters
)


def main():

    print("=" * 70)
    print("SkyVLM LM Training Smoke Test")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SkyVLMForCausalLM(
        vocab_size=16384,
        dim=512,
        num_layers=20,
        num_heads=8,
        ffn_hidden_dim=1408,
        max_seq_len=512
    ).to(device)

    total, trainable = count_parameters(model)

    print(
        f"Parameters: {total:,}"
    )

    print(
        f"Parameters: {total / 1e6:.2f}M"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # --------------------------------------------------------
    # Dummy data
    # --------------------------------------------------------

    batch_size = 8
    seq_len = 256

    input_ids = torch.randint(
        0,
        16384,
        (batch_size, seq_len),
        device=device
    )

    labels = torch.randint(
        0,
        16384,
        (batch_size, seq_len),
        device=device
    )

    # --------------------------------------------------------
    # Memory reset
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Training steps
    # --------------------------------------------------------

    model.train()

    print()
    print("Running training steps...")
    print("-" * 70)

    for step in range(20):

        start = time.time()

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda"
        ):

            logits, loss = model(
                input_ids,
                labels
            )

        loss.backward()

        optimizer.step()

        elapsed = time.time() - start

        if step % 5 == 0:

            print(
                f"Step {step:02d} | "
                f"Loss {loss.item():.4f} | "
                f"{elapsed:.2f}s"
            )

    # --------------------------------------------------------
    # VRAM
    # --------------------------------------------------------

    if device.type == "cuda":

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        print()
        print(
            f"Peak allocated VRAM: "
            f"{peak_allocated:.3f} GB"
        )

        print(
            f"Peak reserved VRAM : "
            f"{peak_reserved:.3f} GB"
        )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    print()
    print("Testing generation...")

    model.eval()

    prompt = torch.randint(
        0,
        16384,
        (1, 10),
        device=device
    )

    with torch.no_grad():

        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            top_k=50
        )

    print(
        f"Prompt shape    : {prompt.shape}"
    )

    print(
        f"Generated shape : {generated.shape}"
    )

    print(
        f"Generated IDs   : "
        f"{generated[0].tolist()}"
    )

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()