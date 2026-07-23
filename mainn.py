from dotenv import load_dotenv

load_dotenv(".env.adv",override=True)

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
   # print(app.invoke(input={"question": "what is agent memory?"}))
    result = app.invoke(
        {
            "question": "who is the winner of FIFA 2026?"
        }
    )

    print(result["generation"])