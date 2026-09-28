"""Private Scheduler endpoint. Cloud Run IAM verifies the caller's OIDC identity."""

import logging
from http import HTTPStatus

from adapters import Compute, GitHub, Lease
from config import Settings
from flask import Flask, jsonify
from fleet import Fleet

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)


@app.post("/reconcile")
def reconcile():
    lease = None
    try:
        settings = Settings.from_env()
        lease = Lease(settings.lease_bucket)
        if not lease.acquire():
            return jsonify(status="locked")
        result = Fleet(settings, GitHub(settings), Compute(settings)).reconcile()
        logging.info("Fleet result: %s", result)
        return jsonify(result)
    except Exception as error:
        # Provider exceptions may contain secrets. Emit only the exception type.
        logging.error("Reconciliation failed: %s", type(error).__name__)
        return jsonify(status="error"), HTTPStatus.SERVICE_UNAVAILABLE
    finally:
        if lease is not None:
            try:
                lease.release()
            except Exception as error:
                logging.error("Lease release failed: %s", type(error).__name__)
