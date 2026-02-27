"""
VitalSense — Periodic Retraining Scheduler
============================================
Runs independently from the web server.
Schedule via cron or run continuously.

Usage:
    python scripts/retrain_scheduler.py              # runs every 24h
    python scripts/retrain_scheduler.py --interval 6 # every 6 hours
    python scripts/retrain_scheduler.py --once       # run once and exit
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(name)s  %(message)s')
logger = logging.getLogger("scheduler")


async def run_once():
    from backend.retrainer import schedule_retraining
    logger.info("▶ Running retraining job...")
    await schedule_retraining()
    logger.info("✓ Done")


async def run_loop(interval_hours: float):
    logger.info(f"🕐 Scheduler started — retraining every {interval_hours}h")
    while True:
        logger.info(f"\n{'─'*50}")
        logger.info(f"  Retraining job @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'─'*50}")
        await run_once()
        logger.info(f"  Next run in {interval_hours}h")
        await asyncio.sleep(interval_hours * 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=24.0, help="Hours between retraining runs")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    if args.once:
        asyncio.run(run_once())
    else:
        asyncio.run(run_loop(args.interval))
