1) Set your_api_key in .env file to your actual OpenAI API key.
2) Run "docker build -f .\Dockerfile -t cover_letter_generator:v1.0 ."
3) Run "docker-compose -f .\docker-compose.yml up -d" for starting the application
4) Run "docker-compose -f .\docker-compose.yml up down" to stop the application