import express from "express";
import * as fs from "fs";
import * as path from 'path';
import * as body_parser from "body-parser";
import {pipeline}  from "@huggingface/transformers"
import { getLlama, LlamaChatSession } from "node-llama-cpp"


const generator = await pipeline("text-generation", "HuggingFaceTB/SmolLM2-1.7B-Instruct");

const message =
  {
    role: "system",
    content: "You are a helpful assistant that helps people find clothing articles based on their preferences. Do not mention anything else other than the clothing article name and description.",
  };

/* const llama = await getLlama();
const model = await llama.loadModel({
  modelPath: path.resolve("./Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"),
});

const context = await model.createContext();
const session = new LlamaChatSession({
  context: context.getSequence(),
  systemPrompt: "You are a helpful assistant that helps people find clothing articles based on their preferences.",
}); */


const app = express();
const port = 8080;

app.use(express.json());
app.use(express.static(path.resolve("./static/html")));



app.post("/start", async (req, res) => {
  if (!req.body.preferences && typeof req.body.preferences !== Array) {
    return res.status(400).json({response: "Missing preferences in request body"});
  }
  let preferences = req.body.preferences.toString();
  let response = await generator([message, {role: "user", content: "Provide the user with a single clothing article based on the preferences provided: " + preferences}], {
    max_new_tokens: 128,
  });
  //let response = await session.prompt("Provide the user with a single clothing article based on the preferences provided: " + preferences);
  return res.status(200).json({response: response[0].generated_text.at(-1).content});

});
app.post("/like", async (req, res) => {
  if (!req.body.article && typeof req.body.article !== String) {
    return res.status(400).send("Missing clothing article name");
  }
  let response = await generator([message, {role: "user", content: "The user liked this article suggestion, keep this in mind and provide another clothing article based on their preferences."}], {
    max_new_tokens: 128,
  });
  //let response = await session.prompt("The user liked this article suggestion, keep this in mind and provide another clothing article based on their preferences.");
  return res.status(200).json({response: response[0].generated_text.at(-1).content});
})
app.post("/dislike", async (req, res) => {
  if (!req.body.article && typeof req.body.article !== String) {
    return res.status(400).send("Missing clothing article name");
  }
  let response = await generator([message, {role: "user", content: "The user disliked this article suggestion, keep this in mind and provide another clothing article based on their preferences."}], {
    max_new_tokens: 128,
  });

  //let response = await session.prompt("The user dislike this article suggestion, keep this in mind and provide another clothing article based on their preferences.");

  return res.status(200).json({response: response[0].generated_text.at(-1).content});
})

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});