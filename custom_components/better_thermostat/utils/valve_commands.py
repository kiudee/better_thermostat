"""Record completed valve writes and invalidate ambiguous delivery."""

from time import time


def record_valve_command(host, entity_id, percent, method):
    """Capture the position only after all required device writes finish."""
    trv = host.real_trvs[entity_id]
    trv.last_valve_percent = int(percent)
    trv.last_valve_method = method
    trv.valve_command_uncertain = False
    manager = getattr(host, "state_mgr", None)
    if manager is not None:
        manager.record_response_command(entity_id, float(percent), time())
        manager.mark_dirty()
    save = getattr(host, "schedule_save_state", None)
    if callable(save):
        save()


def invalidate_valve_command(host, entity_id):
    """Discard an assumed limit after a possibly partial write."""
    trv = host.real_trvs[entity_id]
    trv.last_valve_percent = None
    trv.valve_command_uncertain = True
    trv.valve_write_failures += 1
    trv.last_valve_write_failure = time()
    manager = getattr(host, "state_mgr", None)
    if manager is not None:
        manager.invalidate_response_command(entity_id, trv.last_valve_write_failure)
        manager.mark_dirty()
    save = getattr(host, "schedule_save_state", None)
    if callable(save):
        save()
