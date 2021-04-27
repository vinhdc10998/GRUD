from argparse import ArgumentParser

def get_argument():
    description = 'Genotype Imputation'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--root-dir', type=str, required=True,
                        dest='root_dir', help='Data folder')
    parser.add_argument('--model-config-dir', type=str, required=True,
                        dest='model_config_dir', help='Model config folder')
    parser.add_argument('--model-type', type=str, required=True,
                        dest='model_type', help='Model type')
    parser.add_argument('--gpu', action='store_true',
                        dest='gpu', help='Using GPU')
    parser.add_argument('--batch-size', type=int, default=2, required=False,
                        dest='batch_size', help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, required=False,
                        dest='epochs', help='Epochs')
    parser.add_argument('--regions', type=str, default=1, required=False,
                        dest='regions', help='Region range')
    parser.add_argument('--chr', type=str, default='chr22', required=False,
                        dest='chromosome', help='Chromosome')
    parser.add_argument('--lr', type=float, default=1e-4, required=False,
                        dest='learning_rate', help='Learning rate')
    parser.add_argument('--gamma', type=str, default=0.1, required=False,
                        dest='gamma', help='gamma in loss function')
    parser.add_argument('--output-model-dir', type=str, default='model/weights', required=False,
                        dest='output_model_dir', help='Output weights model dir')
    parser.add_argument('--early-stopping', action='store_true',
                        dest='early_stopping', help='Early stopping')
    parser.add_argument('--model-dir', type=str, default='model/weights', required=False,
                        dest='model_dir', help='Weights model dir')
    parser.add_argument('--result-gen-dir', type=str, default='results/', required=False,
                        dest='result_gen_dir', help='result gen dir')
    parser.add_argument('--best-model', action='store_true',
                        dest='best_model', help='Get best model to eval')
    parser.add_argument('--test-dir', type=str, default='../Dataset/test_100_samples_GT', required=False,
                        dest='test_dir', help='test data folder')

    args = parser.parse_args()
    return args