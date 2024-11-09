import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

class FilePaths:
    """
    Filenames and paths to data
    """
    fn_char_list = '../model/wordCharsList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'

def get_img_height() -> int:
    """
    Fixed height for Neural network.
    """
    return 32

def get_img_size(line_mode: bool=False) -> Tuple[int, int]:
    """
    Height is fixed for NN, width is set according to training mode (single words or text lines)
    """
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def write_summary(average_train_loss: List[float], char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """
    Writing training summary file for NN
    """
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())
    
def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())
    
def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """
    Training neural network
    """
    epoch = 0
    summary_char_error_rates = []
    summary_word_accuracies = []

    train_loss_in_epoch = []
    average_train_loss = []

    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf') # Best validation character error rate
    no_improvement_since = 0 # Number of epochs no improvement of character error rate occurred
    # Early stopping after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # Training
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
            train_loss_in_epoch.append(loss)

        # Validating
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # Writing summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        average_train_loss.append((sum(train_loss_in_epoch)) / len(train_loss_in_epoch))
        write_summary(average_train_loss, summary_char_error_rates, summary_word_accuracies)

        # Reseting train loss list
        train_loss_in_epoch = []

        # If best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {best_char_error_rate}')
            no_improvement_since += 1

        # Stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement for {early_stopping} epochs. Training stopped.')

def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """
    Validating NN
    """
    print('Validating model...')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' %dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')
    
    # Printing validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy

def infer(model: Model, fn_img: Path) -> None:
    """
    Recognizing text in image provided by file path
    """
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')

def parse_args() -> argparse.Namespace:
    """
    Parsing arguments from the command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='wordbeamsearch')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB', action='store_true')
    parser.add_argument('--line_mode', help='Image used for inference', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()

def main():
    """
    Main function.
    """

    # Parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # Training the model
    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

        # When in line mode, take care to have a whitespace in the char list
        char_list = loader.char_list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list
        
        # Saving characters and words
        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))
        
        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, decoder_type)
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # Evaluate it on the validation set
    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        validate(model, loader, args.line_mode)
    
    # Infering text on test image
    elif args.mode == 'infer':
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        infer(model, args.img_file)

if __name__ == '__main__':
    main()
