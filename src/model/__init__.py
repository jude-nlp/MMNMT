# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel  # , TRANSFORMER_LAYER_PARAMS
from .memory import HashingMemory


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # memory
    if params.use_memory:
        HashingMemory.check_params(params)
        s_enc = [x for x in params.mem_enc_positions.split(',') if x != '']
        s_dec = [x for x in params.mem_dec_positions.split(',') if x != '']
        assert len(s_enc) == len(set(s_enc))
        assert len(s_dec) == len(set(s_dec))
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_enc)
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_dec)
        params.mem_enc_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_enc]
        params.mem_dec_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_dec]
        assert len(params.mem_enc_positions) + len(params.mem_dec_positions) > 0
        assert len(params.mem_enc_positions) == 0 or 0 <= min([x[0] for x in params.mem_enc_positions]) <= max([x[0] for x in params.mem_enc_positions]) <= params.n_layers - 1
        assert len(params.mem_dec_positions) == 0 or 0 <= min([x[0] for x in params.mem_dec_positions]) <= max([x[0] for x in params.mem_dec_positions]) <= params.n_layers - 1

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            for x in s:
                print(os.path.isfile(x))
            print(all([x == '' or os.path.isfile(x) for x in s]))

            assert all([x == '' or os.path.isfile(x) for x in s])



def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
    Build model.
    """
    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            # reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            model.load_state_dict(reloaded)

        logger.info("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()
    else:
        # build
        encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings)
            set_pretrain_emb(decoder, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}

                # enc_reload.pop('pred_layer.proj.weight')
                # enc_reload.pop('pred_layer.proj.bias')
                # 这里为了修复torch.Size([2, 1024]) from checkpoint, the shape in current model is torch.Size([3, 1024]) bug
                # enc_reload['lang_embeddings.weight'] = encoder.state_dict()['lang_embeddings.weight']
                # enc_reload['pred_layer.proj.weight'] = encoder.state_dict()['pred_layer.proj.weight']   # 我之前训练的de-en，是没有这个层的，但是现在XLM我又用到了，所以添加上，并且随机初始化
                # enc_reload['pred_layer.proj.bias'] = encoder.state_dict()['pred_layer.proj.bias']   # update
                if params.with_adapter:
                    layer_6 = {}    # 0-6层，第6层
                    for k, v in enc_reload.items():
                        k_list = k.split('.')
                        # 用第5层的参数去初始化第6层
                        if len(k_list[1]) == 1 and int(k_list[1]) == 5:
                            k_list[1] = "6"
                            key_6 = '.'.join(k_list)
                            logger.info("new_key:%s" % key_6)
                            layer_6[key_6] = v
                    for k, v in layer_6.items():
                        enc_reload[k] = v

                encoder.load_state_dict(enc_reload)

            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
                
                for i in range(params.n_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            dec_reload[name % i] = decoder.state_dict()[name % i]

                # 这里为了修复torch.Size([2, 1024]) from checkpoint, the shape in current model is torch.Size([3, 1024]) bug
                # dec_reload['lang_embeddings.weight'] = decoder.state_dict()['lang_embeddings.weight']
                if params.with_adapter:
                    layer_6 = {}    # 0-6层，第6层
                    for k, v in dec_reload.items():
                        k_list = k.split('.')
                        # 用第5层的参数去初始化第6层
                        if len(k_list[1]) == 1 and int(k_list[1]) == 5:
                            k_list[1] = "6"
                            key_6 = '.'.join(k_list)
                            logger.info("new_key:%s" % key_6)
                            layer_6[key_6] = v
                    for k, v in layer_6.items():
                        dec_reload[k] = v

                decoder.load_state_dict(dec_reload)

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))
        # 冻结参数
        # 依据指定超参数冻结
        if params.freeze_encoder_layer_num != -1:
            for k, p in encoder.named_parameters():
                logger.info("%s:%s" % (k, p.requires_grad))
                split = k.split('.')
                # if split[0] == 'position_embeddings' or split[0] == 'lang_embeddings' or split[0] == 'embeddings' or split[0] == 'layer_norm_emb': # 冻结Embedding层
                    # p.requires_grad = False
                    # continue
                if len(split[1]) == 1 and int(split[1]) < params.freeze_encoder_layer_num:
                # if len(split[1]) == 1 and int(split[1]) in [0, 1, 2, 3, 5]:
                    p.requires_grad = False
            logger.info("after set encoder requires_grad to False")
            for k, p in encoder.named_parameters():
                logger.info("%s:%s" % (k, p.requires_grad))
        if params.freeze_decoder_layer_num != -1:
            for k, p in decoder.named_parameters():
                logger.info("%s:%s" % (k, p.requires_grad))
                split = k.split('.')
                if len(split[1]) == 1 and int(split[1]) < params.freeze_decoder_layer_num:
                    p.requires_grad = False
            logger.info("after set decoder requires_grad to False")
            for k, p in decoder.named_parameters():
                logger.info("%s:%s" % (k, p.requires_grad))

        return encoder.cuda(), decoder.cuda()
