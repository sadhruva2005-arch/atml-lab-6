from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-hi"
try:
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    text = "politicians do not have permission to do what needs to be done."
    encoded = tokenizer(text, return_tensors="pt")
    output = model.generate(**encoded)
    print("Pretrained output:", tokenizer.decode(output[0], skip_special_tokens=True))
except Exception as e:
    print("Error:", e)
