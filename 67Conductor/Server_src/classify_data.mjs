import { pipeline, AutoProcessor, AutoModelForImageTextToText, load_image, TextStreamer, AutoModelForImageFeatureExtraction } from "@huggingface/transformers";
import * as fs from "fs";
import * as path from 'path';
const model = "onnx-community/FastVLM-0.5B-ONNX";
const model_extractor = "Xenova/all-MiniLM-L6-v2";
const processor = await AutoProcessor.from_pretrained(model);

const processor_extractor = await AutoProcessor.from_pretrained(model_extractor);
const feature_extractor_model = pipeline("feature-extraction", model_extractor)
const imageTextToTextModel = await AutoModelForImageTextToText.from_pretrained(model, {
  dtype: "q4",
});


//const extractor = await pipeline("feature-extraction", "model_q4")


const message = {
    role: "system",
    content: "You give short and concise descriptions of clothing articles.",
}
const message2 = {
    role: "user",
    content: "<image>Describe this clothing article",
}
const prompt = processor.apply_chat_template([message, message2], {
    add_generation_prompt: true,
});
let img = await load_image(path.resolve("./static/dataset/train/hat/00d94e21-5891-492e-be0e-792e7338c077.jpg"))
console.log("Image loaded");
let inputs = await processor(img, prompt, {
    add_special_tokens: false,
});

const output = await imageTextToTextModel.generate({
    ...
    inputs,
    max_new_tokens: 64,
    length_penalty: -9999999.0,
    do_sample: false,
    streamer: new TextStreamer(processor.tokenizer, {
        skip_prompt: false,
        skip_special_tokens: true,
    }),
});

const decoded = processor.batch_decode(
  outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
  { skip_special_tokens: true },
);
console.log(decoded[0]);