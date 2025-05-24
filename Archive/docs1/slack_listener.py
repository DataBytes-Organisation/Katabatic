import os
import time
import logging
import matplotlib.pyplot as plt
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Slack tokens from environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise ValueError("Both SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set as environment variables.")

# Initialize Slack clients
web_client = WebClient(token=SLACK_BOT_TOKEN)
socket_mode_client = SocketModeClient(app_token=SLACK_APP_TOKEN, web_client=web_client)

def process_command(command, channel_id, user_id):
    """
    Process commands sent by users and respond accordingly.
    """
    if command == "start training":
        send_message(channel_id, "Starting training process... Updates will be sent here. ⏳")
        for step in range(1, 6):  # Simulated training steps
            time.sleep(2)  # Simulate some work
            send_message(channel_id, f"Training progress: Step {step}/5 ⏳")
        send_message(channel_id, "Training complete! 🎉")
    
    elif command == "run evaluation":
        send_message(channel_id, "Starting evaluation process... Please wait. 🧪")
        for step in range(1, 4):  # Simulated evaluation steps
            time.sleep(2)
            send_message(channel_id, f"Evaluation progress: Step {step}/3 🔍")
        send_message(channel_id, "Evaluation complete! ✅ Results are ready.")

    elif command == "show results":
        send_message(channel_id, "Fetching results... 📊")
        upload_chart_to_slack(channel_id)
        send_message(channel_id, "Results:\n- Accuracy: 92%\n- Loss: 0.08\n- Precision: 95%\n- Recall: 90%")

    else:
        send_message(channel_id, f"I'm sorry, I don't recognize the command: '{command}'. Please try again.")

def send_message(channel_id, text):
    """
    Send a message to a specific Slack channel.
    """
    try:
        web_client.chat_postMessage(channel=channel_id, text=text)
    except Exception as e:
        logger.error(f"Failed to send message: {e}")

def generate_chart():
    """
    Generate a chart and save it as a temporary file.
    """
    try:
        # Example data
        metrics = ['Accuracy', 'Loss', 'Precision', 'Recall']
        values = [92, 8, 95, 90]  # Example percentages (Loss is scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_title("Model Performance Metrics")

        chart_path = "results_chart.png"
        plt.savefig(chart_path)
        plt.close(fig)
        logger.info("Chart generated successfully.")
        return chart_path
    except Exception as e:
        logger.error(f"Failed to generate chart: {e}")
        return None

def upload_chart_to_slack(channel_id):
    """
    Generate a chart and upload it to Slack using files_upload.
    """
    chart_path = generate_chart()
    if not chart_path:
        send_message(channel_id, "Failed to generate chart.")
        return

    try:
        response = web_client.files_upload(
            channels=channel_id,
            file=chart_path,
            title="Results Visualization",
        )
        if response["ok"]:
            logger.info("Chart uploaded to Slack successfully.")
    except Exception as e:
        logger.error(f"Failed to upload chart: {e}")
        send_message(channel_id, "Failed to upload chart to Slack.")
    finally:
        if os.path.exists(chart_path):
            os.remove(chart_path)

def handle_event(payload):
    """
    Handle Slack events received via the Socket Mode connection.
    """
    event = payload.get("event", {})
    if event.get("type") == "message" and "subtype" not in event:
        user_id = event.get("user")
        text = event.get("text")
        channel_id = event.get("channel")

        if user_id and text:
            logger.info(f"Message received: '{text}' from user '{user_id}' in channel '{channel_id}'")

            # Ignore the bot's own messages
            bot_id = payload.get("authorizations", [{}])[0].get("user_id")
            if user_id == bot_id:
                logger.info("Ignoring bot's own message.")
                return

            # Process the user command
            process_command(text.lower(), channel_id, user_id)

def handle_socket_mode_request(client: SocketModeClient, request: SocketModeRequest):
    """
    Handle incoming Socket Mode requests.
    """
    if request.type == "events_api":
        response = SocketModeResponse(envelope_id=request.envelope_id)
        client.send_socket_mode_response(response)
        handle_event(request.payload)

def main():
    """
    Main function to initialize the Slack bot and start listening for events.
    """
    logger.info("Tokens seem valid")
    logger.info("Bot is up and running!")

    socket_mode_client.socket_mode_request_listeners.append(handle_socket_mode_request)
    socket_mode_client.connect()

    try:
        # Keep the bot running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        socket_mode_client.disconnect()

if __name__ == "__main__":
    main()
