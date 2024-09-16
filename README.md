The aim of this project was to create a multilingual chatbot that can be able to interact with user in any of the south african
languages which will be a prototype for absa south africa chatbot (abby) which only uses english as a medium of response and interaction.


The project uses a 4 models:
- The LLama model provided by facebook to provide human interaction
- The google translator to translatte to and from the llama model since it limited in south african languages
- The sa language detector which is a model to detect the languages imputed by the user and pass the language code to the google translator( Trained the model myself project on my profile)
- the speech language detector from google and the google transcriber model to convert the recorded message to text before passing it on to the llama model.


The project uses streamlit to deploy the chatbot.
