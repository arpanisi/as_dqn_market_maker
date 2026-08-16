"""Step 11 Telegram and Discord notification channels."""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol


HttpPost = Callable[[str, Mapping[str, object]], None]


class NotificationChannel(Protocol):
    name: str

    def send(self, message: str) -> None:
        ...


def urllib_post_json(url: str, payload: Mapping[str, object]) -> None:
    data = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10):
        return


@dataclass(frozen=True)
class TelegramChannel:
    token: str
    chat_id: str
    http_post: HttpPost = urllib_post_json
    name: str = "telegram"

    def send(self, message: str) -> None:
        token = urllib.parse.quote(self.token, safe="")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.http_post(url, {"chat_id": self.chat_id, "text": message})


@dataclass(frozen=True)
class DiscordChannel:
    webhook_url: str
    http_post: HttpPost = urllib_post_json
    name: str = "discord"

    def send(self, message: str) -> None:
        self.http_post(self.webhook_url, {"content": message})


@dataclass(frozen=True)
class NotificationResult:
    channel: str
    sent: bool
    error: str | None = None


class NotificationManager:
    def __init__(self, channels: list[NotificationChannel], logger: logging.Logger | None = None) -> None:
        self.channels = channels
        self.logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        http_post: HttpPost = urllib_post_json,
        logger: logging.Logger | None = None,
    ) -> "NotificationManager":
        env = env or os.environ
        logger = logger or logging.getLogger(__name__)
        channels: list[NotificationChannel] = []

        telegram_token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
        telegram_chat_id = env.get("TELEGRAM_CHAT_ID", "").strip()
        if telegram_token and telegram_chat_id:
            channels.append(TelegramChannel(telegram_token, telegram_chat_id, http_post=http_post))
            logger.info("notification channel active: telegram")
        else:
            missing = [name for name, value in (("TELEGRAM_BOT_TOKEN", telegram_token), ("TELEGRAM_CHAT_ID", telegram_chat_id)) if not value]
            logger.warning("notification channel disabled: telegram missing %s", ", ".join(missing))

        discord_url = env.get("DISCORD_WEBHOOK_URL", "").strip()
        if discord_url:
            channels.append(DiscordChannel(discord_url, http_post=http_post))
            logger.info("notification channel active: discord")
        else:
            logger.warning("notification channel disabled: discord missing DISCORD_WEBHOOK_URL")

        return cls(channels, logger)

    def startup_validation(self) -> list[NotificationResult]:
        return self.send_all("system starting")

    def send_fill(self, message: str) -> list[NotificationResult]:
        return self.send_all(message)

    def send_hourly_summary(self, message: str) -> list[NotificationResult]:
        return self.send_all(message)

    def send_error(self, message: str) -> list[NotificationResult]:
        return self.send_all(message)

    def send_all(self, message: str) -> list[NotificationResult]:
        results: list[NotificationResult] = []
        for channel in self.channels:
            try:
                channel.send(message)
                results.append(NotificationResult(channel.name, True))
            except Exception as exc:
                self.logger.exception("notification channel failed: %s", channel.name)
                results.append(NotificationResult(channel.name, False, str(exc)))
        return results
