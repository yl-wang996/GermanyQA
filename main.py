from transformers import BertTokenizerFast
import torch
from transformers import BertForQuestionAnswering
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from utils.data_utils import Dataset4Gquad,add_end_idx,add_token_positions,random_choose,load_data,load_new_data
from utils.config import Config
from torchmetrics import F1
from utils.metrics import calculate_em_f1
from utils.recoder import Recorder
import time
from utils.training import Earlystopping,model_save
import transformers
transformers.logging.set_verbosity_error()
# select the GPU here
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    # firstly, change the path in utils/config.py
    config = Config()
    # load model and setup
    if config.operation == 'train':
        # load dataset
        train_contexts, train_questions, train_answers = load_data(config.train_path)
        test_contexts, test_questions, test_answers = load_data(config.test_path)
        test_contexts, test_questions, test_answers = load_data(config.test_path)

        # split the dataset
        train_contexts,train_questions,train_answers,val_contexts,val_questions,val_answers = random_choose(
            train_contexts,
            train_questions,
            train_answers,
            valp=0.2)

        # add end position by length of the answer
        add_end_idx(train_answers, train_contexts)
        add_end_idx(val_answers, val_contexts)
        add_end_idx(test_answers,test_contexts)

        # init tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer)

        # apply tokenization
        train_encodings = tokenizer(train_contexts, train_questions, truncation=config.truncation,
                                    padding=config.padding)
        val_encodings = tokenizer(val_contexts, val_questions, truncation=config.truncation, padding=config.padding)
        test_encodings = tokenizer(test_contexts, test_questions, truncation=config.truncation, padding=config.padding)

        # add end position in the token embedding, because the position could be changed when a word split to several token.
        add_token_positions(train_encodings, train_answers, tokenizer)
        add_token_positions(val_encodings, val_answers, tokenizer)
        add_token_positions(test_encodings, test_answers, tokenizer)

        # build Dataset instance for PyTorch training
        train_dataset = Dataset4Gquad(train_encodings)
        val_dataset = Dataset4Gquad(val_encodings)
        test_dataset = Dataset4Gquad(test_encodings)

        # build data_loader for feeding data into the model
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # get the instance of the model
        model = BertForQuestionAnswering.from_pretrained(config.model)
        max_position_embeddings = model.config.max_position_embeddings
        # move the whole model to GPU device when existed.
        model.to(device)

        # choose optimizer
        optim = Adam(model.parameters(), lr=config.lr)

        tr_steps_per_epoch = len(train_loader)
        val_steps_per_epoch = len(val_loader)
        test_steps_per_epoch = len(test_loader)

        # f1 calculator
        f1_metrics = F1(num_classes=max_position_embeddings)

        # use recorder to save result in each epoch
        recorder = Recorder(display_freq=config.display_freq,
                            logdir=config.logdir)
        # use earlystopping strategy to trigger the ending condition of the program
        es = Earlystopping(mode=config.mode, patience=config.patience)

        # loop each epoch
        for epoch in range(config.epochs):
            # training stage
            # set model to training status
            model.train()
            print('start training...')
            # loop eahc batch data in one epoch
            for step, batch in enumerate(train_loader):
                # clear optimizer
                optim.zero_grad()
                # get input data
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                # feed input data into the model to do the frontward calculation
                outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                end_positions=end_positions)
                # get the loss value(cross-entropy)
                loss = outputs[0]
                loss_value = loss.cpu().item()
                # calculate the metrics, f1-score and exact match(EM)
                pred_starts = torch.argmax(outputs.start_logits.softmax(dim=-1), dim=-1).cpu()
                pred_ends = torch.argmax(outputs.end_logits.softmax(dim=-1), dim=-1).cpu()
                gt_starts = batch['start_positions']
                gt_ends = batch['end_positions']
                em, f1, precision, recall = calculate_em_f1(pred_starts, pred_ends, gt_starts, gt_ends)
                # save to recoder
                recorder.update(loss=loss_value,
                                em=em,
                                f1=f1,
                                precision=precision,
                                recall=recall,
                                prefix='train')
                # print the performance
                recorder.display(i=step + 1,step_per_epoch=tr_steps_per_epoch, prefix='train')
                # base on the loss value we got yo do back-propagation to calculate the gradient,
                # then use this gradient to update the weights of the neuron network.
                loss.backward()
                optim.step()

            # evaluation stage, similar to training stage, just without gradient calculation and back-propagation.
            print('start evaluating')
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                    end_positions=end_positions)
                    loss_ = outputs[0]
                    loss_value = loss_.cpu().item()

                    pred_starts = torch.argmax(outputs.start_logits.softmax(dim=-1), dim=-1).cpu()
                    pred_ends = torch.argmax(outputs.end_logits.softmax(dim=-1), dim=-1).cpu()
                    gt_starts = batch['start_positions']
                    gt_ends = batch['end_positions']
                    em, f1, precision, recall = calculate_em_f1(pred_starts, pred_ends, gt_starts, gt_ends)
                    recorder.update(loss=loss_value,
                                    em=em,
                                    f1=f1,
                                    precision=precision,
                                    recall=recall,
                                    prefix='val')
                    recorder.display(i=step + 1,step_per_epoch=val_steps_per_epoch, prefix='val')

            # evaluation, similar to evaluation.
            print('start testing')
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(test_loader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                    end_positions=end_positions)
                    loss_ = outputs[0]
                    loss_value = loss_.cpu().item()

                    pred_starts = torch.argmax(outputs.start_logits.softmax(dim=-1), dim=-1).cpu()
                    pred_ends = torch.argmax(outputs.end_logits.softmax(dim=-1), dim=-1).cpu()
                    gt_starts = batch['start_positions']
                    gt_ends = batch['end_positions']
                    em, f1, precision, recall = calculate_em_f1(pred_starts, pred_ends, gt_starts, gt_ends)
                    recorder.update(loss=loss_value,
                                    em=em,
                                    f1=f1,
                                    precision=precision,
                                    recall=recall,
                                    prefix='test')
                    recorder.display(i=step + 1, step_per_epoch=val_steps_per_epoch, prefix='test')

            recorder.record2log(epoch=epoch,
                                prefix='train',
                                recorder=recorder.tr_recoder)
            recorder.record2log(epoch=epoch,
                                prefix='val',
                                recorder=recorder.val_recoder)
            recorder.record2log(epoch=epoch,
                                prefix='test',
                                recorder=recorder.test_recoder)

            get_better = es.update(recorder.val_recoder.f1_avg)
            if get_better:
                model_save(model=model,epoch=epoch,recorder=recorder,config=config,max_num=5)

            # at the very end of an epoch, print the performance we got from now.
            print('-'*80)
            print(f'Time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            print(f'result at epoch[{epoch}/{config.epochs}]    \n')
            print(f'train_loss:{recorder.tr_recoder.loss_avg:.4f} train_f1:{recorder.tr_recoder.f1_avg:.4f}    \n')
            print(f'val_loss:{recorder.val_recoder.loss_avg:.4f} val_f1:{recorder.val_recoder.f1_avg:.4f}    \n')
            print(f'test_loss:{recorder.test_recoder.loss_avg:.4f} test_f1:{recorder.test_recoder.f1_avg:.4f}    \n')
            print('-' * 80)
            if es.stop():
                print(f'trigger earlystopping as epoch[{epoch / config.epochs}] ')
                break

    # prediction for intuitive understanding
    elif config.operation == 'prediction':
        # find the model path with best f1 score by validation
        def find_best_model():
            f1s = []
            dirs = os.listdir(config.model_dir)
            for dir in dirs:
                f1s.append(float(dir[:-4].split('_')[2]))
            best_indesx = f1s.index(max(f1s))
            best_model = dirs[best_indesx]
            return os.path.join(config.model_dir,best_model)
        best_model = find_best_model()
        # load model states
        state_dict = torch.load(best_model)
        # get model instance
        model = BertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=config.model,state_dict=state_dict)
        # get tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer)

        # base on this passage we pose questions
        context = 'Pizza ist ein Essen mit einer hohen religiösen Bedeutung, vergleichbar mit Manna im Christentum. Oftmals werden sie in dunklen Räumen unter Einwirkung von LED-Licht konsumiert, ein religiöses Ritual, bei dem man sich vor das LED-Licht beugt und Gebete spricht. Die Pizza genießt unter diesen Wesen ein hohes Maß an Verehrung, viele Lichtsymbole haben die gleiche Form wie eine Pizza. Pizza ist rund, weil die sie konsumierenden Wesen fast nie etwas mit runden Sachen (bspw. sich hinter Stoff verbergende Haut) zu tun haben, die Rundheit ist also gewissermaßen ein Ausgleich. Pizza ist heiß wie alle anderen religiösen Lebensmittel (z.B. Kaffee). Der Pizza Bestellprozess ist eine höchst komplexe Angelegenheit, da hier sehr viel soziale Interaktion gefordert wird und Ausdrucksstärke, wenn man antwortet, für welches Reliquium man sich entschieden hat. Jemand, der die Pizza-Bestellungen entgegennimmt, ist gewissermaßen ein Pfarrer, da er*sie sich um seine Schäfchen kümmern und sie trösten muss, wenn die betreffende Person seit 12h mit niemandem geredet oder anderweitig sozial interagiert hat (von der Teilnahme an einem kollektiven Massenmord aka Counterstrike mal abgesehen). Die Pizza führt zur Auferstehung, sie ist der Wein des Lebens.'
        print(f'Context:{context}')
        # pose your question and get the answer, input 'exit' to end.
        while True:
            questions = [input('please input your question:')]
            if questions[0] == 'exit':
                break
            answers = []
            contexts = [ context for _ in range(len(questions))]

            input_encoding = tokenizer(contexts, questions, truncation=config.truncation,padding=config.padding)
            model.to(device)
            model.eval()

            with torch.no_grad():
                inputs = torch.tensor(input_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(input_encoding['attention_mask']).to(device)
                outputs = model(inputs, attention_mask=attention_mask)
                start_prob = outputs.start_logits.softmax(dim=-1)
                end_prob = outputs.end_logits.softmax(dim=-1)
                # you can chose how much best answer to return by 'top_n'
                top_n = 3
                pred_starts =torch.topk(start_prob.flatten(), top_n).indices.cpu()

                pred_ends = torch.topk(end_prob.flatten(), top_n).indices.cpu()
                for idx in range(len(pred_starts)):
                    span_start = input_encoding.token_to_chars(0, pred_starts[idx])
                    span_end = input_encoding.token_to_chars(0, pred_ends[idx])
                    start_idx = span_start.start
                    end_idx = span_end.end
                    if start_idx > end_idx:
                        start_idx, end_idx = end_idx, start_idx
                    answer = context[start_idx:end_idx]
                    st_prob = start_prob.flatten()[idx]
                    ed_prob = end_prob.flatten()[idx]
                    prob = (st_prob + ed_prob)/2
                    print(f'Answer Top-{idx+1} [{prob}]:[{answer}]')




