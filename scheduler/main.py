from pytorch_workload import create_workload


def main():
    model, x = create_workload()

    print(model)
    print("\nInput shape:", x.shape)

    y = model(x)

    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()
