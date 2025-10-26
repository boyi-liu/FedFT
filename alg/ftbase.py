import copy
import os
import math

from transformers import TrainingArguments, Trainer
from alg.base import BaseClient, BaseServer
from datasets import load_dataset
from utils.model_utils import load_model


class FTBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        _, self.tokenizer = load_model(args)
        self.training_args = TrainingArguments(
            output_dir=f"./client{self.id}",  # where to save the output log
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epoch,
            logging_steps=10,  # gap of steps between two logging
            save_steps=50000,
            save_total_limit=2,
            fp16=True,
            optim="adamw_torch"
        )
        self.load_data()
        self.lora = {}

    def load_data(self):
        train_dir = os.path.join('./dataset', self.args.dataset, f'train/{self.id}.json')
        test_dir = os.path.join('./dataset', self.args.dataset, f'test/{self.id}.json')

        self.dataset = load_dataset("json", data_files={'train': train_dir, 'test': test_dir})
        self.dataset['train'] = self.dataset['train'].map(self.format_example)
        self.dataset['test'] = self.dataset['test'].map(self.format_example)

    def format_example(self, example):
        prompt = f"Instruct: {example['question']}\nAnswer:"
        return {
            "input_ids": self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0],
            "labels": self.tokenizer(example["answer"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0]
        }

    def run(self, model):
        client_model = copy.deepcopy(model)
        client_model.train()

        Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            processing_class=self.tokenizer,
        ).train()

        self.lora = {k: v for k, v in client_model.state_dict().items() if "lora_" in k}

    def local_test(self, model):
        model.eval()

        trainer = Trainer(
            model=model,
            args=self.training_args,
            eval_dataset=self.dataset["test"],
            processing_class=self.tokenizer
        )
        metrics = trainer.evaluate()
        if "eval_loss" in metrics and metrics["eval_loss"] is not None and not math.isnan(metrics["eval_loss"]):
            metrics["perplexity"] = float(math.exp(metrics["eval_loss"]))
        else:
            metrics["perplexity"] = float("inf")

        print(f"Client {self.id} local test metrics:", metrics)
        return metrics

class FTBaseServer(BaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.model, _ = load_model(args)
        self.client_models = []

        self.round = 0

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def sample(self):
        pass

    def local_run(self):
        for client in self.clients: client.run(self.model)

    def aggregate(self):
        data_sum = sum([len(client.dataset['train']) for client in self.clients])
        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)

        for client in self.clients:
            model = client.lora
            for k, v in model.items():
                aggregated[k] = aggregated[k] + v * len(client.dataset['train']) / data_sum

        self.model.load_state_dict(aggregated, strict=False)
        print("Aggregated model updated.")

    def test_all(self):
        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.model)
            all_metrics.append(metrics)


        avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
        avg_perplexity = sum(m["perplexity"] for m in all_metrics) / len(all_metrics)
        return {'loss': avg_loss, 'perplexity': avg_perplexity}