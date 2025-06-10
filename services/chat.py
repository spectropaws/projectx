from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path = "./unsloth.Q8_0.gguf",
        temperature=0.7,
        max_tokens=256,
        top_p=0.95,
        n_ctx=2048,
        n_threads=8,
        verbose=True,
)

response = llm("Set up a static IP and DNS")
print(response)
