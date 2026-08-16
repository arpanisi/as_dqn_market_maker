import logging

from notifications.channels import NotificationManager


def test_notification_manager_disables_missing_channels_without_crashing(caplog):
    caplog.set_level(logging.WARNING)

    manager = NotificationManager.from_env(env={}, http_post=lambda url, payload: None)

    assert manager.channels == []
    assert "telegram missing TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID" in caplog.text
    assert "discord missing DISCORD_WEBHOOK_URL" in caplog.text


def test_startup_validation_sends_to_both_configured_channels():
    sent = []

    def fake_post(url, payload):
        sent.append((url, payload))

    manager = NotificationManager.from_env(
        env={
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "DISCORD_WEBHOOK_URL": "https://discord.example/webhook",
        },
        http_post=fake_post,
    )

    results = manager.startup_validation()

    assert [result.channel for result in results] == ["telegram", "discord"]
    assert all(result.sent for result in results)
    assert sent[0][1]["text"] == "system starting"
    assert sent[1][1]["content"] == "system starting"


def test_channel_failures_are_independent():
    def fake_post(url, payload):
        if "telegram" in url:
            raise RuntimeError("telegram down")

    manager = NotificationManager.from_env(
        env={
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "DISCORD_WEBHOOK_URL": "https://discord.example/webhook",
        },
        http_post=fake_post,
    )

    results = manager.send_fill("fill")

    assert results[0].channel == "telegram"
    assert not results[0].sent
    assert results[1].channel == "discord"
    assert results[1].sent
