import numpy as np


def get_mse(code_A, code_B):
    return ((code_A - code_B) ** 2).mean()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_A_path", type=str,
                        help="Path to code A")
    parser.add_argument("--code_B_path", type=str,
                        help="Path to code B")
    hps = parser.parse_args()

    code_A = np.load(hps.code_A_path)
    code_B = np.load(hps.code_B_path)

    code_A_train = code_A.item()['train']
    code_A_test = code_A.item()['test']
    code_B_train = code_B.item()['train']
    code_B_test = code_B.item()['test']

    code_mse_train = get_mse(code_A_train, code_B_train)
    code_mse_train_last = get_mse(code_A_train[:, -768:], code_B_train[:, -768:])
    code_mse_test = get_mse(code_A_test, code_B_test)
    code_mse_test_last = get_mse(code_A_test[:, -768:], code_B_test[:, -768:])

    print("Train Code MSE: {}".format(code_mse_train))
    print("Train Last Code MSE: {}".format(code_mse_train_last))
    print("Test Code MSE: {}".format(code_mse_test))
    print("Test Last Code MSE: {}".format(code_mse_test_last))
