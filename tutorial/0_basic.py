#在本脚本中，您将学习
#1.如何将huggingface数据集实用程序集成到openprompt中，以在不同的数据集中实现快速学习
#2.如何使用模板语言
#3.实例化模板。模板如何将输入示例包装为模板示例
#4.我们如何隐藏PLM tokenization细节，并提供一个简单的tokenization
#5.如何使用一个或者多个标签词
#6.构造描述器。如何像传统的预训练模型一样训练提示。

device_ids=[2,3]

# load dataset
from datasets import load_dataset
raw_dataset = load_dataset('super_glue', 'cb')
# raw_dataset['train'][0]
from datasets import load_from_disk
# raw_dataset = load_from_disk("/home/hushengding/huggingface_datasets/saved_to_disk/super_glue.cb")
# Note that if you are running this scripts inside a GPU cluster, there are chances are you are not able to connect to huggingface website directly.
# In this case, we recommend you to run `raw_dataset = load_dataset(...)` on some machine that have internet connections.
# Then use `raw_dataset.save_to_disk(path)` method to save to local path.
# Thirdly upload the saved content into the machiine in cluster.
# Then use `load_from_disk` method to load the dataset.

from openprompt.data_utils import InputExample

dataset = {}
for split in ['train', 'validation', 'test']:
    dataset[split] = []
    for data in raw_dataset[split]:
        input_example = InputExample(text_a = data['premise'], text_b = data['hypothesis'], label=int(data['label']), guid=data['idx'])
        dataset[split].append(input_example)
print(dataset['train'][0])

# You can load the plm related things provided by openprompt simply by calling:
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")

# Constructing Template
# 可以从 yaml 配置构造模板，但也可以通过直接传递参数构造模板
from openprompt.prompts import ManualTemplate
template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

#为了更好地理解模板如何包装示例，我们将一个实例可视化

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)


# 现在，包装好的示例已准备好传递给标记器，从而生成语言模型的输入。
# 您可以使用标记器自己标记输入，但我们建议使用包装标记器，它是InputExample的包装标记器尾部。
# 如果您使用“load_plm”函数，则已给出包装器，否则，应根据配置选择合适的包装器in `openprompt.plms.__init__.py`.
# 注意，当 t5用于分类时，我们只需要传递 < pad > < tra _ id _ 0 > < eos > 来解码。
# 损失计算为<extra_id_0>。因此，通过decoder_max_length=3节省了空间
wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
# or
from openprompt.plms import T5TokenizerWrapper
wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")

# You can see what a tokenized example looks like by
tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
print(tokenized_example)
print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))

# Now it's time to convert the whole dataset into the input format!
# Simply loop over the dataset to achieve it!

model_inputs = {}
for split in ['train', 'validation', 'test']:
    model_inputs[split] = []
    for sample in dataset[split]:
        tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
        model_inputs[split].append(tokenized_example)


# 我们提供了一个“PromptDataLoader”类来帮助您完成上述所有事项，并将它们包装成一个“torch.DataLoader风格的迭代器”
from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
# next(iter(train_dataloader))


# Define the verbalizer
# 在分类中，您需要定义描述器，它是从词汇表上的逻辑到最终标签概率的映射。让我们来看看描述器的详细信息：

from openprompt.prompts import ManualVerbalizer
import torch

# 例如，描述器在每个类中包含多个标签词
myverbalizer = ManualVerbalizer(tokenizer, num_classes=3,
                        label_words=[["yes","sure"], ["no","not"], ["maybe","probably","perhaps"]])

print(myverbalizer.label_words_ids)
logits = torch.randn(2,len(tokenizer)) # 从plm创造一个伪输出看哪个verbalizer起作用
print("process_logits:",myverbalizer.process_logits(logits))


# 尽管您可以手动将plm、模板和描述器组合在一起，但我们提供了一个管道模型，它从PromptDataLoader中获取批处理数据并生成类逻辑
from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda(device=device_ids[0])

# Now the training is standard
from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

for epoch in range(10):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda(device=device_ids[0])
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

# Evaluate
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

allpreds = []
alllabels = []
for step, inputs in enumerate(validation_dataloader):
    if use_cuda:
        inputs = inputs.cuda(device=device_ids[0])
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print(acc)

