"""
Train the model aided by backtranslation.
Currently, only the standard Skip-Thought setting
(prediect prev and next sentences) is supported.
"""

import argparse
import os
from pprint import pprint

import torch
import yaml
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm

from models import basic, SkipThought
from utils import DataReader, Preprocessor, Vocab


def train(args):
    vocab = Vocab.load(args.vocab, max_size=args.vocab_size)
    data_reader = DataReader(data_dir=args.data_dir, shuffle=True)
    preprocessor = Preprocessor(
        predict_prev=args.predict_prev,
        predict_cur=args.predict_cur,
        predict_next=args.predict_next,
        vocab=vocab, max_length=args.max_length, gpu=args.gpu)
    model = SkipThought(
        rnn_type=args.rnn_type, num_words=len(vocab),
        word_dim=args.word_dim, hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        predict_prev=args.predict_prev,
        predict_cur=args.predict_cur,
        predict_next=args.predict_next)
    print(model)

    if args.pretrained is not None:
        print(f'Loading pretrained model from {args.pretrained}')
        model.load_state_dict(
            torch.load(args.pretrained,
                       map_location=lambda storage, loc: storage))
    if args.gpu > -1:
        model.cuda(args.gpu)
    optimizer = optim.Adam(model.parameters())

    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'log'))

    def add_scalar_summary(name, value, step):
        summary_writer.add_scalar(tag=name, scalar_value=value,
                                  global_step=step)

    def add_text_summary(name, value, step):
        summary_writer.add_text(tag=name, text_string=value,
                                global_step=step)

    def variable(tensor, volatile=False):
        return Variable(tensor, volatile=volatile)

    def run_train_iter(batch):
        if not model.training:
            model.train()
        src, tgt = preprocessor(batch)
        src = (variable(src[0]), src[1])
        for k in tgt:
            tgt[k] = (variable(tgt[k][0]), tgt[k][1])
        logits = model.forward(src=src, tgt=tgt)
        loss = 0
        for k in tgt:
            logits_k = logits[k]
            tgt_k = tgt[k]
            loss = loss + basic.sequence_cross_entropy(
                logits=logits_k[:-1], targets=tgt_k[0][1:],
                length=tgt_k[1] - 1)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=10)
        optimizer.step()
        return loss.data[0]

    def ids_to_words(ids):
        words = []
        eos_id = vocab.stoi(vocab.eos)
        for id_ in ids:
            words.append(vocab.itos(id_))
            if id_ == eos_id:
                break
        return words

    def generate_using_decoder(name, src, max_length):
        _, encoder_state = model.encoder(words=src[0], length=src[1])
        if isinstance(encoder_state, tuple):  # LSTM
            encoder_state = encoder_state[0]
        context = (encoder_state.transpose(0, 1).contiguous()
                   .view(-1, args.hidden_dim))
        batch_size = src[1].size(0)

        bos_id = vocab.stoi(vocab.bos)
        bos = Variable(src[1].new(1, batch_size).fill_(bos_id))
        decoder = model.get_decoder(name)
        prev_pred = bos
        done = torch.zeros(batch_size).byte()
        hyps = []
        prev_state = context.unsqueeze(0)
        for t in range(max_length):
            if done.all():
                break
            decoder_input = prev_pred
            logit, prev_state = decoder(words=decoder_input,
                                        prev_state=prev_state)
            pred = logit.max(2)[1]
            prev_pred = pred
            hyps.append(pred.data)
        hyps = torch.cat(hyps, dim=0).transpose(0, 1).tolist()
        return hyps

    def generate(batch):
        # Greedy search
        src, tgt = preprocessor(batch)
        src = (variable(src[0]), src[1])
        for k in tgt:
            tgt[k] = (variable(tgt[k][0], volatile=True), tgt[k][1])
        batch_size = src[0].size(1)
        max_length = src[0].size(0) * 2
        generated = {}
        for k in tgt:
            generated[k] = generate_using_decoder(
                name=k, src=src, max_length=max_length)
        results = []
        for i in range(batch_size):
            res = {'src': ' '.join(ids_to_words(src[0][:src[1][i], i].data)),
                   'tgt': {},
                   'out': {}}
            for k in tgt:
                res['tgt'][k] = ' '.join(ids_to_words(tgt[k][0][1:, i].data))
                res['out'][k] = ' '.join(ids_to_words(generated[k][i]))
            results.append(res)
        return results

    def generate_synthetic_batch(real_batch):
        def sort_by_length(tgt_of_key):
            sorted_length, sort_inds = tgt_of_key[1].sort(
                dim=0, descending=True)
            return tgt_of_key[0][:, sort_inds], sorted_length

        # Forward: given prev, generate cur'
        _, tgt = preprocessor(real_batch)
        tgt_prev, tgt_prev_length = sort_by_length(tgt['prev'])
        syn_src_fw = generate_using_decoder(
            name='next',
            src=(variable(tgt_prev[1:], volatile=True),
                 tgt_prev_length - 1),
            max_length=args.max_length)
        # Backward: given next, generate cur''
        tgt_next, tgt_next_length = sort_by_length(tgt['next'])
        syn_src_bw = generate_using_decoder(
            name='prev',
            src=(variable(tgt_next[1:], volatile=True),
                 tgt_next_length - 1),
            max_length=args.max_length)
        syn_batch_fw = []
        syn_batch_bw = []
        for i in range(len(real_batch)):
            syn_src_fw_str = ' '.join(ids_to_words(syn_src_fw[i]))
            syn_src_bw_str = ' '.join(ids_to_words(syn_src_bw[i]))
            syn_batch_fw.append(
                (real_batch[i][0], syn_src_fw_str, real_batch[i][2]))
            syn_batch_bw.append(
                (real_batch[i][0], syn_src_bw_str, real_batch[i][2]))
        return syn_batch_fw, syn_batch_bw

    global_step = 0

    def print_samples():
        model.eval()
        num_samples = 2
        samples = data_reader.next_batch(size=num_samples, peek=True)
        syn_samples_fw, syn_samples_bw = generate_synthetic_batch(samples)
        gen_results = generate(samples)
        syn_gen_results_fw = generate(syn_samples_fw)
        syn_gen_results_bw = generate(syn_samples_bw)
        text_val = ''
        for i, res in enumerate(gen_results):
            text_val += f'* sample (real) #{i}\n'
            text_val += f'\t* src: {res["src"]}\n'
            for k in res['tgt']:
                tgt_k = res['tgt'][k]
                out_k = res['out'][k]
                text_val += f'\t* {k} (tgt): {tgt_k}\n'
                text_val += f'\t* {k} (out): {out_k}\n'
        for i, res in enumerate(syn_gen_results_fw):
            text_val += f'* sample (syn_fw) #{i}\n'
            text_val += f'\t* src: {res["src"]}\n'
            for k in res['tgt']:
                tgt_k = res['tgt'][k]
                out_k = res['out'][k]
                text_val += f'\t* {k} (tgt): {tgt_k}\n'
                text_val += f'\t* {k} (out): {out_k}\n'
        for i, res in enumerate(syn_gen_results_bw):
            text_val += f'* sample (syn_bw) #{i}\n'
            text_val += f'\t* src: {res["src"]}\n'
            for k in res['tgt']:
                tgt_k = res['tgt'][k]
                out_k = res['out'][k]
                text_val += f'\t* {k} (tgt): {tgt_k}\n'
                text_val += f'\t* {k} (out): {out_k}\n'
        add_text_summary('Sample', value=text_val, step=global_step)

    for epoch in range(args.max_epoch):
        data_reader.start_epoch()
        for batch in tqdm(data_reader.iterator(args.batch_size),
                          desc=f'Epoch {epoch}'):
            # Train on real batch
            real_loss = run_train_iter(batch)
            # Train on synthetic batches
            syn_batch_fw, syn_batch_bw = generate_synthetic_batch(batch)
            syn_loss_fw = run_train_iter(syn_batch_fw)
            syn_loss_bw = run_train_iter(syn_batch_bw)
            global_step += 1
            add_scalar_summary(name='real_loss', value=real_loss,
                               step=global_step)
            add_scalar_summary(name='syn_loss_fw', value=syn_loss_fw,
                               step=global_step)
            add_scalar_summary(name='syn_loss_bw', value=syn_loss_bw,
                               step=global_step)
            if global_step % args.print_every == 0:
                print_samples()
            if global_step % args.save_every == 0:
                model_filename = f'model-{global_step}.pt'
                model_path = os.path.join(args.save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                print(f'\nIter #{global_step}: '
                      f'Saved checkpoint to {model_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--vocab-size', type=int, default=20000)
    parser.add_argument('--max-length', type=int, default=30)
    parser.add_argument('--rnn-type', default='lstm')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--predict-prev', default=False, action='store_true')
    parser.add_argument('--predict-cur', default=False, action='store_true')
    parser.add_argument('--predict-next', default=False, action='store_true')
    parser.add_argument('--word-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--print-every', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained', default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir)
    config = {'model': {'rnn_type': args.rnn_type,
                        'bidirectional': args.bidirectional,
                        'word_dim': args.word_dim,
                        'hidden_dim': args.hidden_dim,
                        'predict_prev': args.predict_prev,
                        'predict_cur': args.predict_cur,
                        'predict_next': args.predict_next},
              'train': {'batch_size': args.batch_size,
                        'vocab_size': args.vocab_size,
                        'max_length': args.max_length}}
    pprint(config)

    config_path = os.path.join(args.save_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    train(args)


if __name__ == '__main__':
    main()
