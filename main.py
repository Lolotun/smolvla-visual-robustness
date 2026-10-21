from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
def main():    

    policy = SmolVLAPolicy.from_pretrained(
        'HuggingFaceVLA/smolvla_libero'
    )
    policy.to('cuda')
    policy.eval()

    print('SmolVLA loaded successfully!')


if __name__ == "__main__":
    main()
