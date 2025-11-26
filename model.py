# 12. Save trained adapter + tokenizer + label encoder
# Save PEFT adapter (this saves only adapter weights)
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)                 # saves adapter + config (PEFT)
tokenizer.save_pretrained(output_dir)             # save tokenizer
joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))  # label encoder
print("Saved model and artifacts to", output_dir)
