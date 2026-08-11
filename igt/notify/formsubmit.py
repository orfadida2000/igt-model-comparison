import io
import logging
import time
import traceback
import zipfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager

import requests
from email_validator import EmailNotValidError, validate_email

from igt.typing import StrPathLike
from igt.utils.io import normalize_path


def validate_normalize_email(
    email_address: str,
    *,
    check_deliverability: bool = True,
) -> str:
    """
    Validates an email address, checks DNS deliverability (if check_deliverability is True),
    and returns the safely normalized version.
    """
    try:
        # check_deliverability=True performs the DNS lookup to verify MX records exist
        email_info = validate_email(email_address, check_deliverability=check_deliverability)

        # Return the clean, normalized string
        return email_info.normalized

    except EmailNotValidError as e:
        # Catch the specific validation error and raise a standard ValueError
        # so your notification system can catch and handle it cleanly.
        raise ValueError(f"Invalid or undeliverable email: {e}") from e


def send_formsubmit_email_notification(
    formsubmit_id: str,
    subject: str,
    message: str,
    success: bool = True,
    status: str | None = None,
    duration_seconds: float | None = None,
    script_name: str | None = None,
    file_paths: Sequence[StrPathLike] | None = None,
    zip_filename: str | None = None,
) -> dict:
    """
    Sends an email notification via FormSubmit.
    If file_paths is provided and not empty, it zips the files and attaches the archive.
    """

    ajax_url = f"https://formsubmit.co/ajax/{formsubmit_id}"
    upload_url = f"https://formsubmit.co/{formsubmit_id}"

    # Spoof your Live Server domain to bypass FormSubmit's security lock
    headers = {
        "Origin": "http://127.0.0.1",
        "Referer": "http://127.0.0.1",
        "Accept": "application/json",
    }

    if duration_seconds is None or duration_seconds < 0:
        duration_string = "N/A"
    else:
        elapsed_seconds = round(duration_seconds)
        days, remainder = divmod(elapsed_seconds, 86_400)
        hours, remainder = divmod(remainder, 3_600)
        minutes, seconds = divmod(remainder, 60)

        if days:
            duration_string = (
                f"{days} day{'s' if days != 1 else ''}, {hours:02d}:{minutes:02d}:{seconds:02d}"
            )
        else:
            duration_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Inject FormSubmit specific configurations directly into the payload
    payload = {
        "Script Name": script_name if script_name else "N/A",
        "Status": status if status else "N/A",
        "Duration": duration_string,
        "File Count": str(len(file_paths or [])),
        "Success": "Yes" if success else "No",
        "Message": message,
        "_subject": subject,
        "_captcha": "false",
        "_template": "table",
    }

    try:
        # Condition 1: No files provided or the list is empty
        if not file_paths:
            response = requests.post(
                ajax_url,
                headers=headers,
                json=payload,
                timeout=10,
            )

        # Condition 2: The list has files
        else:
            file_paths = [
                normalize_path(fp, parameter_name="file_path_attachment") for fp in file_paths
            ]

            logging.info("Zipping %d file(s) in memory...", len(file_paths))
            zip_buffer = io.BytesIO()

            # Create the ZIP archive in memory
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_archive:
                for file_path in file_paths:
                    # Path(file_path).name reliably extracts the file name without the full directory structure
                    zip_archive.write(file_path, arcname=file_path.name)

            # Reset the buffer's position to the beginning so 'requests' can read it
            zip_buffer.seek(0)

            zip_filename = zip_filename if zip_filename else "attachments.zip"
            clean_zip_filename = (
                zip_filename.strip().replace(" ", "_").lower().rstrip(".zip") + ".zip"
            )  # Ensure the filename is clean and ends with .zip
            files = {"attachment": (clean_zip_filename, zip_buffer, "application/zip")}

            response = requests.post(
                upload_url,
                headers=headers,
                data=payload,
                files=files,
                timeout=10,
            )

        if response.status_code == 200:
            try:
                response_data = response.json()
                response_data["success"] = (
                    True if response_data.get("success", "false").lower() == "true" else False
                )
                response_data["status_code"] = (
                    response.status_code
                )  # Include the status code for reference
                response_data["response_text"] = (
                    None  # Include the key for consistency, even if it's None
                )

                return response_data
            except ValueError:
                return {
                    "success": False,
                    "message": f"Server returned non-JSON response. Status: {response.status_code}",
                    "status_code": response.status_code,
                    "response_text": response.text,
                }
        else:
            return {
                "success": False,
                "message": f"Server rejected request. Status: {response.status_code}",
                "status_code": response.status_code,
                "response_text": response.text,
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Exception occurred: {e}",
            "status_code": None,
            "response_text": None,
        }


def send_formsubmit_email_script_success_notification(
    formsubmit_id: str,
    message: str,
    *,
    duration_seconds: float | None = None,
    script_name: str | None = None,
    file_paths: Sequence[StrPathLike] | None = None,
    zip_filename: str | None = None,
) -> dict:
    """
    Sends a success notification email via FormSubmit.
    """

    return send_formsubmit_email_notification(
        formsubmit_id=formsubmit_id,
        subject="Success Notification",
        message=message,
        success=True,
        status="Success",
        duration_seconds=duration_seconds,
        script_name=script_name,
        file_paths=file_paths,
        zip_filename=zip_filename,
    )


def send_formsubmit_email_script_error_notification(
    formsubmit_id: str,
    message: str,
    *,
    duration_seconds: float | None = None,
    script_name: str | None = None,
    file_paths: Sequence[StrPathLike] | None = None,
    zip_filename: str | None = None,
) -> dict:
    """
    Sends an error notification email via FormSubmit.
    """

    return send_formsubmit_email_notification(
        formsubmit_id=formsubmit_id,
        subject="Error Notification",
        message=message,
        success=False,
        status="Error",
        duration_seconds=duration_seconds,
        script_name=script_name,
        file_paths=file_paths,
        zip_filename=zip_filename,
    )


@contextmanager
def error_email_notifier(
    formsubmit_id: str | None,
    *,
    script_name: str | None = None,
    start_counter: float | None = None,
) -> Generator[None]:
    """
    A context manager that runs a block of code. If an exception occurs
    (other than KeyboardInterrupt or SystemExit), it sends an email with
    the traceback and then re-raises the error.
    """

    if formsubmit_id is None:
        # If no FormSubmit ID is provided, just yield control without sending notifications
        yield
        return

    start_counter = start_counter if start_counter is not None else time.perf_counter()

    try:
        # yield gives control back to the block of code inside the 'with' statement
        yield
    except Exception:
        end_counter = time.perf_counter()
        elapsed_time = end_counter - start_counter

        # Get the full error traceback as a string
        error_details = traceback.format_exc()

        message = f"Error Details:\n{error_details}"

        send_formsubmit_email_script_error_notification(
            formsubmit_id=formsubmit_id,
            message=message,
            duration_seconds=elapsed_time,
            script_name=script_name,
        )

        # Re-raise the error so the script still halts and prints the traceback to your terminal
        raise
