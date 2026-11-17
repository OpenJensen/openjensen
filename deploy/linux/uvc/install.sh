#!/bin/bash
# Register the camera quirk with DKMS. The distribution module is restored by dkms remove.
set -euo pipefail
src=$(cd -- "$(dirname -- "$0")" && pwd)
package=openjensen-uvc
version=1.0
dest=/usr/src/$package-$version

if (( EUID != 0 )); then
    echo "Run this driver installer with sudo." >&2
    exit 1
fi
for tool in dkms depmod modinfo make python3; do
    command -v "$tool" >/dev/null || { echo "Install $tool before setting up the camera driver." >&2; exit 1; }
done
if [[ ${1-} == --offline ]]; then
    export OPENJENSEN_UVC_OFFLINE=1
    shift
fi
if (( $# == 0 )); then
    set -- "$(uname -r)"
    next=$(readlink -e /boot/vmlinuz || true)
    if [[ ${next##*/} == vmlinuz-* ]]; then
        next=${next##*/vmlinuz-}
        [[ $next == "$1" ]] || set -- "$1" "$next"
    fi
fi
for kernel in "$@"; do
    [[ $kernel =~ ^[a-zA-Z0-9][a-zA-Z0-9._+-]*$ && -f /lib/modules/$kernel/build/Makefile ]] || {
        echo "Install matching kernel headers for $kernel." >&2; exit 1;
    }
    current=$(modinfo -k "$kernel" -F filename uvcvideo)
    case "$current" in
        */kernel/drivers/media/usb/uvc/uvcvideo.ko*) ;;
        */updates/*|*/extra/*)
            dkms status -m "$package" -v "$version" -k "$kernel" | grep -q ': installed' || {
                echo "Another UVC driver override already owns $kernel ($current)." >&2; exit 1;
            };;
        *) echo "A modular stock UVC driver is required for $kernel ($current)." >&2; exit 1;;
    esac
done

added=false
created=false
rollback() {
    status=$?
    if $added; then dkms remove "$package/$version" --all || exit "$status"; fi
    if $created; then rm -f -- "$dest/build_driver.py" "$dest/dkms.conf"; rmdir "$dest" || true; fi
    exit "$status"
}
trap rollback ERR
if [[ -e $dest ]]; then
    cmp "$src/build_driver.py" "$dest/build_driver.py"
    cmp "$src/dkms.conf" "$dest/dkms.conf"
else
    mkdir -m 755 "$dest"
    created=true
    install -m 755 "$src/build_driver.py" "$dest/build_driver.py"
    install -m 644 "$src/dkms.conf" "$dest/dkms.conf"
fi
if [[ -z $(dkms status -m "$package" -v "$version") ]]; then
    dkms add "$package/$version"
    added=true
fi
for kernel in "$@"; do
    if ! dkms status -m "$package" -v "$version" -k "$kernel" | grep -q ': installed'; then
        dkms install "$package/$version" -k "$kernel"
    fi
    modinfo -k "$kernel" -F parm uvcvideo | grep -q '^mjpeg_bandwidth_cap:'
done
trap - ERR
echo "Camera driver installed. Reboot or close camera users and reload uvcvideo to activate it."
