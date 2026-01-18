FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install torch numpy

# Copy the new submission file and the brain
# (We need the .pth file so the brain works!)
COPY submission.py .
COPY connect4_brain.pth .

# IMPORTANT: -u is critical for the text protocol to work through Docker!
CMD ["python", "-u", "submission.py"]