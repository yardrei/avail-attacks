import click
import math

import datetime
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from avail_attacks.dataloaders.grad_explode_loader import DataloaderExploder

from avail_attacks.train import *
from avail_attacks.models import *
from avail_attacks.dataloaders.noise import DataloaderWithNoise
from avail_attacks.pgd.pgd import DataloaderPGD
from avail_attacks.pgd.sampler import SamplesSelectionMethodology
from avail_attacks.dataloaders.different_label import (
    DataloaderDifferentLabel,
    WrongSelectionMethodology,
)
from avail_attacks.dataloaders.white_images import DataloaderWhite
from avail_attacks.utils import DatasetChoice
import matplotlib.pyplot as plt


@click.command()
@click.option("--dataset", default="mnist", help="The dataset to use")
@click.option("--model-number", default=1, help="The model to use")
@click.option("--epochs", default=5, help="Number of epochs")
@click.option("--lr", default=1e-3, help="The default learning rate")
@click.option("--use-noise-dl", default=False, help="Use noisy dataloader baseline")
@click.option("--use-white-images", default=False, help="Use white images")
@click.option("--use-wrong-label", default=False, help="Use the wrong label for images")
@click.option(
    "--wrong-label-method",
    default=1,
    help="The method for the wrong label dataloader",
)
@click.option(
    "--use-grad-exploder",
    default=False,
    is_flag=True,
    help="Use grad exploder in updates",
)
@click.option("--use-pgd", default=False, help="Use pgd")
@click.option("--use-cw", default=False, help="Use Carlini-Wagner")
@click.option("--cw-c-constant", type=float, default=10.0)
@click.option(
    "--pgd-samples-selection",
    help="Only required if using pgd. Determines how mutated samples are selected",
    default="take_from_next_epoch",
)
@click.option(
    "--pgd-targeted",
    type=int,
    default=0,
    help=(
        "Only required if using pgd / cw. Is the attack targeted. "
        "0 = not targeted. "
        "1 = targeted. First optimize images over original label, then pick label at random. "
        "2 = targeted. First optimize images over original label, then pick label with lowest pred. "
        "3 = targeted. Pick a different label at random, and optimize images over that label. Then return adversarial images with the wrong label. "
        "4 = targeted. Pick a different label at random, and optimize images over that label. Then return adversarial images with the original label. "
    ),
)
@click.option("--pgd-step-size", default=0.05, help="The step size in pgd")
@click.option("--change-fraction", default=0.01, help="The percentage of affected data")
@click.option("--attack-epochs", default=30, help="The number of epochs in the attack")
@click.option(
    "--no-dropout",
    is_flag=True,
    default=False,
    help="Whether to use dropout in the model or not",
)
@click.option(
    "--num-idlg-batches",
    default=1,
    help="Replaces noise_fraction for IDLG. Absolute number.",
)
@click.option(
    "--use-grad-clipping", is_flag=True, default=False, help="Whether to clip gradients"
)
@click.option(
    "--grad-clipping-value", default=2, help="The value of gradient clippping"
)
@click.option("--black-box", type=bool, default=False)
@click.option("--transfer-learning", type=int, default=0, help="Use transfer learning")
@click.option(
    "--run-mul-times",
    type=bool,
    default=False,
    help="Prints a graph of multiple fractions",
)
@click.option(
    "--upper-fraction-limit",
    type=float,
    default=0,
    help="Upper limit of the fraction of changes",
)
@click.option(
    "--lower-fraction-limit",
    type=float,
    default=0,
    help="Lower limit of the fraction of changes",
)
@click.option(
    "--number-of-fractions",
    type=int,
    default=0,
    help="Number of fractions between the upper and lower limits",
)
@click.option(
    "--number-of-tests", type=int, default=0, help="Number of test in each fraction"
)
def main(
    dataset: str,
    model_number: int,
    epochs: int,
    lr: float,
    use_noise_dl: bool,
    use_white_images: bool,
    use_wrong_label: bool,
    wrong_label_method: str,
    change_fraction: float,
    use_grad_exploder: bool,
    use_pgd: bool,
    use_cw: bool,
    cw_c_constant: bool,
    pgd_samples_selection: str,
    pgd_targeted: int,
    pgd_step_size: float,
    attack_epochs: int,
    no_dropout: bool,
    num_idlg_batches: int,
    use_grad_clipping: bool,
    grad_clipping_value: float,
    black_box: bool,
    transfer_learning: bool,
    run_mul_times: bool,
    upper_fraction_limit: float,
    lower_fraction_limit: float,
    number_of_fractions: int,
    number_of_tests: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    print(f"Run settings:\ndataset {dataset} epochs {epochs}")
    if use_noise_dl:
        print("Using noise")
    elif use_white_images:
        print("Using white images")
    elif use_wrong_label:
        print(f"Wrong label attack in method {wrong_label_method}")
    elif use_pgd:
        print(
            "PGD attack",
            f"targeted: {pgd_targeted}",
            pgd_samples_selection,
            f"\nNumber of epochs {attack_epochs}",
        )
    elif use_cw:
        print(
            "CW attack",
            f"targeted: {pgd_targeted}",
            pgd_samples_selection,
            f"\nNumber of epochs {attack_epochs}",
        )
    elif use_grad_exploder:
        print("Gradient exploder attack")
    elif black_box:
        print("Nes attack")
    else:
        print("Training the model normally")

    if transfer_learning:
        print("Using a surrogate model")

    if not run_mul_times:
        run_attack(
            dataset=dataset,
            model_number=model_number,
            epochs=epochs,
            lr=lr,
            use_noise_dl=use_noise_dl,
            use_white_images=use_white_images,
            use_wrong_label=use_wrong_label,
            wrong_label_method=wrong_label_method,
            change_fraction=change_fraction,
            use_grad_exploder=use_grad_exploder,
            use_pgd=use_pgd,
            use_cw=use_cw,
            cw_c_constant=cw_c_constant,
            pgd_samples_selection=pgd_samples_selection,
            pgd_targeted=pgd_targeted,
            pgd_step_size=pgd_step_size,
            attack_epochs=attack_epochs,
            no_dropout=no_dropout,
            num_idlg_batches=num_idlg_batches,
            use_grad_clipping=use_grad_clipping,
            grad_clipping_value=grad_clipping_value,
            black_box=black_box,
            transfer_learning=transfer_learning,
            device=device,
        )
    else:
        number_of_runs = number_of_fractions * number_of_tests
        print(f"Running for {number_of_runs} times")
        result_array = np.zeros((number_of_fractions, number_of_tests))

        fractions_array = np.linspace(
            lower_fraction_limit, upper_fraction_limit, number_of_fractions
        )

        for i in range(number_of_fractions):
            for j in range(number_of_tests):
                print(
                    f"Run number {i * number_of_tests + j + 1} out of {number_of_runs}"
                )
                results = run_attack(
                    dataset=dataset,
                    model_number=model_number,
                    epochs=epochs,
                    use_noise_dl=use_noise_dl,
                    use_white_images=use_white_images,
                    use_wrong_label=use_wrong_label,
                    wrong_label_method=wrong_label_method,
                    change_fraction=fractions_array[i],
                    use_grad_exploder=use_grad_exploder,
                    use_pgd=use_pgd,
                    use_cw=use_cw,
                    cw_c_constant=cw_c_constant,
                    pgd_samples_selection=pgd_samples_selection,
                    pgd_targeted=pgd_targeted,
                    pgd_step_size=pgd_step_size,
                    attack_epochs=attack_epochs,
                    no_dropout=no_dropout,
                    num_idlg_batches=num_idlg_batches,
                    use_grad_clipping=use_grad_clipping,
                    grad_clipping_value=grad_clipping_value,
                    black_box=black_box,
                    transfer_learning=transfer_learning,
                    device=device,
                )
                result_array[i][j] = results.per_epoch[-1].test_acc

        median_array = np.median(result_array, axis=1)
        min_array = result_array.min(axis=1)
        max_array = result_array.max(axis=1)

        fig, ax = plt.subplots()
        ax.fill_between(fractions_array, min_array, max_array, alpha=0.5, linewidth=0)
        ax.plot(fractions_array, median_array, linewidth=2)
        plt.savefig("example_plot.png")
        plt.show()


def run_attack(
    dataset: str,
    model_number: int,
    epochs: int,
    lr: float,
    use_noise_dl: bool,
    use_white_images: bool,
    use_wrong_label: bool,
    wrong_label_method: str,
    change_fraction: float,
    use_grad_exploder: bool,
    use_pgd: bool,
    use_cw: bool,
    cw_c_constant: bool,
    pgd_samples_selection: str,
    pgd_targeted: int,
    pgd_step_size: float,
    attack_epochs: int,
    no_dropout: bool,
    num_idlg_batches: int,
    use_grad_clipping: bool,
    grad_clipping_value: float,
    black_box: bool,
    transfer_learning: bool,
    device: torch.device,
):
    assert any(dataset.lower() == member.name for member in DatasetChoice)
    dataset_enum = DatasetChoice[dataset.lower()]

    if model_number == 2:
        train_trans = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        train_trans = transforms.ToTensor(),
        # transforms.Compose([
        # transforms.ToTensor(),
        # # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        test_trans = transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        
        

    # Dataset
    if dataset_enum == DatasetChoice.mnist:
        training_set = datasets.MNIST(
            root="./train_data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        test_set = datasets.MNIST(
            root="./test_data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        number_of_classes = 10

        # Model
        model = VGGMnist(no_dropout=no_dropout).to(device)
        if transfer_learning:
            dataset_model = VGGMnist(no_dropout=no_dropout).to(device)
            dataset_optimizer = optim.Adam(dataset_model.parameters(), lr=lr)
        else:
            dataset_model = model
            dataset_optimizer = None
    else:
        training_set = datasets.CIFAR10(
            root="./train_data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        test_set = datasets.CIFAR10(
            root="./test_data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        number_of_classes = 10

        # Model
        if model_number == 1:
            model = VGGCifar(no_dropout=no_dropout).to(device)
        elif model_number == 2:
            model = ResNet(number_of_classes).to(device)
        else:
            print("Not the right option for cifar model")
            exit(1)

        if transfer_learning:
            dataset_model = VGGCifar(no_dropout=no_dropout).to(device)
            dataset_optimizer = optim.Adam(dataset_model.parameters())
        else:
            dataset_model = model
            dataset_optimizer = None

    batch_size = 128

    # Cross entropy as the loss function
    # using 'sum' instead of 'mean' is critical if the attacking strategy involves
    # adding tiny batches (very few samples), since those samples would then be weighed
    # the same as a full batch
    reduction = "mean"
    if use_grad_exploder:
        reduction = "mean"
    else:
        # scale grad clipping
        # grad_clipping_value = grad_clipping_value * batch_size
        print()
    loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    # Dataloader
    train_dl = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    test_dl = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    sample_selection: SamplesSelectionMethodology = SamplesSelectionMethodology[
        pgd_samples_selection
    ]

    if use_noise_dl:
        n_noisy_samples: int = math.ceil(len(training_set) * change_fraction)
        print(f"Constructing noisy dataloader with {n_noisy_samples} noise samples")
        train_dl = DataloaderWithNoise(
            base_dl=train_dl, n_noisy_samples=n_noisy_samples
        )

    if black_box:
        assert use_pgd, "bbox only implemented in PGD"

    if use_pgd or use_cw:
        train_dl = DataloaderPGD(
            training_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            model=dataset_model,
            loss_func=loss_fn,
            sample_selection_methodology=sample_selection,
            epochs=attack_epochs,
            step_size=pgd_step_size,
            fraction_affected_samples=change_fraction,
            is_targeted=pgd_targeted,
            number_of_classes=number_of_classes,
            use_cw=use_cw,
            cw_c_constant=cw_c_constant,
            is_black_box=black_box,
            transfer_learning=transfer_learning,
            transfer_optimizer=dataset_optimizer,
        )

    if use_wrong_label:
        train_dl = DataloaderDifferentLabel(
            training_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            sample_selection_methodology=WrongSelectionMethodology(wrong_label_method),
            model=dataset_model,
            loss_func=loss_fn,
            fraction_affected_samples=change_fraction,
            number_of_classes=number_of_classes,
            transfer_learning=transfer_learning,
            transfer_optimizer=dataset_optimizer,
        )

    if use_white_images:
        train_dl = DataloaderWhite(
            training_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            fraction_affected_samples=change_fraction,
            number_of_classes=number_of_classes,
        )

    if use_grad_exploder:
        train_dl = DataloaderExploder(
            training_set,
            model,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            use_grad_exploder=use_grad_exploder,
            num_idlg_batches=num_idlg_batches,
            sample_selection=sample_selection,
            loss_fn=loss_fn,
        )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate decay
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)

    model, results = train(
        model,
        train_dl,
        test_dl,
        epochs,
        loss_fn,
        optimizer,
        lr_scheduler,
        device,
        use_grad_clipping,
        grad_clipping_value,
    )

    if use_pgd or use_cw:
        print("Value Counts:")
        for value, count in train_dl.count_dict.items():
            print(f"{value}: {count}", end=" ")
        print()

        print("Predict Value Counts:")
        for value, count in sorted(train_dl.count_dict_predict.items()):
            print(f"{value}: {count}", end=" ")
        print()

    print(f"Final test accuracy: {results.per_epoch[-1].test_acc:.2f}")

    def prettify(l):
        return " ".join(map(lambda x: f"{x:<7,.2f}", l))

    accs = prettify(
        [epoch_res.test_acc for epoch_res in results.per_epoch],
    )
    num_adv_samples = prettify(
        [epoch_res.num_adv_samples for epoch_res in results.per_epoch],
    )
    adv_samples_l2_norm = prettify(
        [epoch_res.mean_l2_of_adv_samples for epoch_res in results.per_epoch],
    )
    grad_norms = prettify(
        [epoch_res.avg_grad_norm for epoch_res in results.per_epoch],
    )
    adv_grad_norms = prettify(
        [epoch_res.adv_avg_grad_norm for epoch_res in results.per_epoch],
    )
    epochs = prettify(list(range(len(results.per_epoch))))
    print(f"per Epoch:                {epochs}")
    print(f"test accuracies:          {accs}")
    print(f"num adv samples:          {num_adv_samples}")
    print(f"mean l2 of adv samples:   {adv_samples_l2_norm}")
    print(f"avg grad norms:           {grad_norms}")
    print(f"adv avg grad norms:       {adv_grad_norms}")

    df = results.to_df()
    ts = datetime.datetime.now().isoformat()
    os.makedirs("results", exist_ok=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    results_csv_path = f"results/results_{ts}_{slurm_job_id}.csv".replace(
        ":", "-"
    )  # doesnt matter if slurm id none
    df.to_csv(results_csv_path)
    results_plot_path = f"results/results_{ts}_{slurm_job_id}.png".replace(":", "-")
    print(f"Results saved to {results_csv_path}")

    df["test_acc"].plot()
    plt.savefig(results_plot_path)
    print(f"Plot saved to    {results_plot_path}")

    return results


if __name__ == "__main__":
    main()
