build:
	docker build -t cover_letter_generator:v1.0 .

run:
	docker-compose up -d

stop:
	docker-compose down

restart:
	make stop
	make run

