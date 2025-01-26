# main.py

# Important Instructions:
# 1. Close any existing Chrome instances.
# 2. Start Chrome with remote debugging enabled:
#    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
# 3. Run the FastAPI server:
#    uvicorn main:app --host 127.0.0.1 --port 8888 --reload --workers 1
# Make sure you set OPENAI_API_KEY=yourOpenAIKeyHere to .env file

import os
os.environ["PYDANTIC_V1_COMPAT_MODE"] = "true"

from langchain_openai import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
import platform
import asyncio
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from browser_use.browser.browser import Browser, BrowserConfig
import logging
import traceback
from datetime import datetime
from typing import List, Optional
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware

import requests
import io
from PyPDF2 import PdfReader


# ----------------------------
# 1. Configure Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# 2. Load Environment Variables
# ----------------------------
load_dotenv()

# Verify the OpenAI API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in .env file. Make sure your .env file is set up correctly."
    )

# ----------------------------
# 3. Initialize FastAPI App
# ----------------------------
app = FastAPI(title="AI Agent API with BrowserUse", version="1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development: allow all origins. In production, specify exact origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 4. Define Pydantic Models
# ----------------------------
class TaskRequest(BaseModel):
    task: str
    url: Optional[str] = None  # Optional URL for doc/Google Drive PDF

class TaskResponse(BaseModel):
    result: str

class TaskStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskRecord(BaseModel):
    id: int
    task: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None  # Duration in seconds
    result: Optional[str] = None
    error: Optional[str] = None

# ----------------------------
# 5. Initialize Task Registry
# ----------------------------
task_records: List[TaskRecord] = []
task_id_counter: int = 0
task_lock = asyncio.Lock()  # To manage concurrent access to task_records

# ----------------------------
# 6. Define Utility Functions
# ----------------------------
def fetch_document_text(url: str) -> str:
    """
    Fetches the document from a given URL (could be PDF or text).
    Returns the extracted text (or raw text if not PDF).
    Raises an exception on any error.
    """
    response = requests.get(url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()

    # If PDF, parse with PyPDF2
    if "pdf" in content_type:
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        # Treat as text/document of some kind
        return response.text

# ----------------------------
# 7. Browser Path Helper
# ----------------------------
def get_chrome_path() -> str:
    """
    Returns the most common Chrome executable path based on the operating system.
    Raises:
        FileNotFoundError: If Chrome is not found in the expected path.
    """
    system = platform.system()
    
    if system == "Windows":
        # Common installation path for Windows
        chrome_path = os.path.join(
            os.environ.get("PROGRAMFILES", "C:\\Program Files"),
            "Google\\Chrome\\Application\\chrome.exe"
        )
    elif system == "Darwin":
        # Common installation path for macOS
        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    elif system == "Linux":
        # Common installation path for Linux
        chrome_path = "/usr/bin/google-chrome"
    else:
        raise FileNotFoundError(f"Unsupported operating system: {system}")
    
    # Verify that the Chrome executable exists at the determined path
    if not os.path.exists(chrome_path):
        raise FileNotFoundError(f"Google Chrome executable not found at: {chrome_path}")
    
    return chrome_path

# ----------------------------
# 8. Define Background Task Function
# ----------------------------
async def execute_task(task_id: int, task: str, url: Optional[str] = None):
    """
    Background task to execute the AI agent.
    Initializes a new browser instance for each task to ensure isolation.
    Optionally fetches and summarizes an external doc/PDF if 'url' is provided.
    """
    global task_records
    browser = None  # Initialize browser instance for this task
    try:
        logger.info(f"Starting background task ID {task_id}: {task}")
        
        # Create and add the task record with status 'running'
        async with task_lock:
            task_record = TaskRecord(
                id=task_id,
                task=task,
                status=TaskStatus.RUNNING,
                start_time=datetime.utcnow()
            )
            task_records.append(task_record)
        
        # If a URL is provided, fetch and summarize its contents
        if url:
            logger.info(f"Task ID {task_id}: Fetching document from URL: {url}")
            try:
                doc_text = fetch_document_text(url)
                # Summarize the doc_text using a secondary LLM call (for brevity, direct call here)
                logger.info(f"Task ID {task_id}: Summarizing document content.")
                summary_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
                doc_summary = summary_llm.predict(
                    "Please summarize the following text:\n" + doc_text
                )
                # Append the summary to the main task to be used as context
                task += f"\n\nAdditional document summary:\n{doc_summary}"
                logger.info(f"Task ID {task_id}: Document content summarized and appended to task.")
            except Exception as doc_error:
                # Log any errors related to doc fetching/summarizing but continue the main task
                logger.error(f"Task ID {task_id}: Error fetching/summarizing doc: {doc_error}")
                logger.error(traceback.format_exc())
                # We can optionally append an error note to the task
                task += f"\n\n[Note: Error reading doc from {url} - {doc_error}]"

        # Initialize a new browser instance for this task
        logger.info(f"Task ID {task_id}: Initializing new browser instance.")
        browser = Browser(
            config=BrowserConfig(
                chrome_instance_path=get_chrome_path(),
                disable_security=True,
                headless=False
            )
        )
        logger.info(f"Task ID {task_id}: Browser initialized successfully.")
        
        # Initialize and run the Agent with the new browser instance
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model="gpt-4o", api_key=api_key),
            browser=browser
        )
        logger.info(f"Task ID {task_id}: Agent initialized. Running task.")
        result = await agent.run()
        logger.info(f"Task ID {task_id}: Agent.run() completed successfully.")
        
        # Update the task record with status 'completed'
        async with task_lock:
            for record in task_records:
                if record.id == task_id:
                    record.status = TaskStatus.COMPLETED
                    record.end_time = datetime.utcnow()
                    record.duration = (record.end_time - record.start_time).total_seconds()
                    record.result = result
                    break

    except Exception as e:
        logger.error(f"Error in background task ID {task_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update the task record with status 'failed'
        async with task_lock:
            for record in task_records:
                if record.id == task_id:
                    record.status = TaskStatus.FAILED
                    record.end_time = datetime.utcnow()
                    record.duration = (record.end_time - record.start_time).total_seconds()
                    record.error = str(e)
                    break
    finally:
        # Ensure that the browser is closed in case of failure or success
        if browser:
            try:
                logger.info(f"Task ID {task_id}: Closing browser instance.")
                await browser.close()
                logger.info(f"Task ID {task_id}: Browser instance closed successfully.")
            except Exception as close_e:
                logger.error(f"Task ID {task_id}: Error closing browser: {close_e}")
                logger.error(traceback.format_exc())

# ----------------------------
# 9. Define POST /run Endpoint
# ----------------------------
@app.post("/run", response_model=TaskResponse)
async def run_task_post(request: TaskRequest, background_tasks: BackgroundTasks):
    """
    POST Endpoint to run the AI agent with a specified task.
    
    - **task**: The task description for the AI agent.
    - **url** (optional): A link to a doc or Google Drive PDF to be fetched, summarized,
      and included as additional context.
    """
    global task_id_counter
    task = request.task
    url = request.url
    logger.info(f"Received task via POST: {task}")
    if url:
        logger.info(f"Optional doc URL provided: {url}")
    
    # Increment task ID
    async with task_lock:
        task_id_counter += 1
        current_task_id = task_id_counter
    
    # Enqueue the background task
    background_tasks.add_task(execute_task, current_task_id, task, url)
    
    # Respond immediately
    return TaskResponse(result="Task is being processed.")

# ----------------------------
# 10. Define GET /run Endpoint
# ----------------------------
@app.get("/run", response_model=TaskResponse)
async def run_task_get(
    task: str = Query(..., description="The task description for the AI agent."),
    background_tasks: BackgroundTasks = None
):
    """
    GET Endpoint to run the AI agent with a specified task.
    
    - **task**: The task description for the AI agent.
    """
    global task_id_counter
    logger.info(f"Received task via GET: {task}")
    
    # Increment task ID
    async with task_lock:
        task_id_counter += 1
        current_task_id = task_id_counter
    
    # Enqueue the background task (no URL support for GET endpoint per instructions)
    background_tasks.add_task(execute_task, current_task_id, task)
    
    # Respond immediately
    return TaskResponse(result="Task is being processed.")

# ----------------------------
# 11. Define GET /lastResponses Endpoint
# ----------------------------
@app.get("/lastResponses", response_model=List[TaskRecord])
async def get_last_responses(
    limit: Optional[int] = Query(100, description="Maximum number of task records to return"),
    status: Optional[TaskStatus] = Query(None, description="Filter by task status")
):
    """
    GET Endpoint to retrieve the last task responses.
    
    - **limit**: The maximum number of task records to return (default: 100).
    - **status**: (Optional) Filter tasks by status ('running', 'completed', 'failed').
    
    Returns a list of task records in descending order of task ID.
    """
    async with task_lock:
        filtered_tasks = task_records.copy()
        if status:
            filtered_tasks = [task for task in filtered_tasks if task.status == status]
        # Sort and limit
        sorted_tasks = sorted(filtered_tasks, key=lambda x: x.id, reverse=True)[:limit]
        return sorted_tasks

# ----------------------------
# 12. Define Root Endpoint
# ----------------------------
@app.get("/")
def read_root():
    return {
        "message": (
            "AI Agent API with BrowserUse is running. Use the /run endpoint with a 'task' field "
            "in the POST request body or as a query parameter in a GET request to execute tasks."
        )
    }

# ----------------------------
# 13. Entry Point
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8888, reload=True, workers=1)