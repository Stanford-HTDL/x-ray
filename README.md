# X-Ray

This repository contains Python code for performing computer vision tasks using Celery with RabbitMQ as the message broker. It utilizes PyTorch and other geospatial imagery processing tools to process the received messages. The results are returned using Redis with Celery, and are also uploaded to Google Cloud Storage. The code is designed to be run in Docker containers using Docker Compose.

## Features

- Asynchronous message processing using Celery and RabbitMQ.
- Computer vision models powered by PyTorch for advanced image processing tasks.
- Integration with Redis for efficient result retrieval.
- Uploading of results to Google Cloud Storage for storage and sharing.
- Containerized application using Docker Compose for easy deployment.

## Prerequisites

Make sure you have the following dependencies installed on your system:

- Docker
- Docker Compose

## Getting Started

To run the application, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the repository's directory.

### Configuration

Before running the application, you need to configure the necessary environment variables. Rename the provided `.env.example` file to `.env` and update the values according to your setup.

The following environment variables need to be configured:

- `RABBITMQ_HOST`: RabbitMQ host address.
- `RABBITMQ_PORT`: RabbitMQ port number.
- `RABBITMQ_USER`: RabbitMQ username.
- `RABBITMQ_PASSWORD`: RabbitMQ password.
- `REDIS_HOST`: Redis host address.
- `REDIS_PORT`: Redis port number.
- `REDIS_PASSWORD`: Redis password.
- `GCS_BUCKET_NAME`: Google Cloud Storage bucket name.
- `GCS_CREDENTIALS_FILE`: Path to the Google Cloud Storage credentials file.

### Build and Run

1. Open a terminal and navigate to the repository's directory.
2. Run the following command to build and start the Docker containers:

   ```
   docker-compose up --build
   ```

   This command will build the required Docker images and start the containers.

3. Once the containers are up and running, the application will be ready to receive messages and process them using the computer vision models.

## Usage

To send a message for processing, you can use any client that can publish messages to RabbitMQ. The message should contain the necessary parameters for the computer vision task.

The application will process the received messages asynchronously using Celery and PyTorch. The results will be stored in Redis and uploaded to Google Cloud Storage. You can retrieve the results by querying the appropriate endpoints or accessing the files in Google Cloud Storage.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please submit a pull request or open an issue.

## License

[BSD 3-Clause License](https://opensource.org/license/bsd-3-clause/)

## Contact

For any inquiries or support, please contact [Richard Correro](mailto:richard@richardcorrero.com). Developed by [Richard Correro](mailto:richard@richardcorrero.com).

