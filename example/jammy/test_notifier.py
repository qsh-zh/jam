from jammy.utils.notifier import jam_notifier

if __name__ == "__main__":
    jam_notifier.notify("sending from jam", "JAM_NOTIFIER Test")
    jam_notifier.notify("second send from jam", "JAM_NOTIFIER Test")
