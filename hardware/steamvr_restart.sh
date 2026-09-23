#!/usr/bin/env bash
# SteamVR 清理与常驻启动脚本
#
# 作用:
#   1. 杀掉上次退出残留的整棵幽灵启动树 (reaper / AppId=250820 / vrstartup /
#      steamvr_room_setup / vrserver / vrmonitor)，避免 `game already running`
#   2. `steam steam://rungameid/250820` 常驻启动完整栈 (vrserver + vrmonitor)
#   3. 轮询等待直到 vrserver 和 vrmonitor 同时就绪后打印确认
#
# 用法:
#   bash hardware/steamvr_restart.sh            # 清理 + 常驻启动 + 等待就绪
#   bash hardware/steamvr_restart.sh --kill     # 只清理幽灵树，不启动
#
# 说明: 需在宿主机终端执行 (沙箱内 pgrep/kill 看不到宿主机进程)。
set -u

STEAMVR_APPID="250820"
# 幽灵启动树匹配模式: reaper 包装 / Steam 启动器 / 房间设置弹窗 / vrstartup
GHOST_PATTERN='AppId=250820|SteamLaunch|steam-launch-wrapper|vrstartup|steamvr_room_setup|srt-bwrap|pv-adverb'
WAIT_TIMEOUT=60          # 等待就绪最长秒数
SELF_PID=$$

log()  { printf '%s\n' "$*"; }
warn() { printf '[!] %s\n' "$*" >&2; }

# ---------- 1. 清理幽灵进程树 ----------
kill_ghosts() {
    log "==> 清理 SteamVR 幽灵进程树 ..."

    # 先温和 TERM 精确进程 (vrserver/vrmonitor) 与幽灵树, 再对残留 KILL -9
    local pids
    # -x 精确匹配 vrserver/vrmonitor; -f 全命令行匹配幽灵树
    pids="$(pgrep -x vrserver; pgrep -x vrmonitor; pgrep -f "$GHOST_PATTERN")"
    # 去重 + 排除自身/父进程, 防止脚本自杀
    pids="$(printf '%s\n' "$pids" | grep -E '^[0-9]+$' | sort -un \
            | grep -v -E "^(${SELF_PID}|${PPID})$" || true)"

    if [ -z "$pids" ]; then
        log "    无残留进程，跳过"
        return 0
    fi

    log "    发现进程: $(printf '%s ' $pids)"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 2

    # 复查仍存活的, 强制杀掉
    local left
    left="$(pgrep -x vrserver; pgrep -x vrmonitor; pgrep -f "$GHOST_PATTERN")"
    left="$(printf '%s\n' "$left" | grep -E '^[0-9]+$' | sort -un \
            | grep -v -E "^(${SELF_PID}|${PPID})$" || true)"
    if [ -n "$left" ]; then
        log "    仍存活, kill -9: $(printf '%s ' $left)"
        # shellcheck disable=SC2086
        kill -9 $left 2>/dev/null || true
        sleep 1
    fi

    # 最终确认
    if pgrep -x vrserver >/dev/null || pgrep -x vrmonitor >/dev/null \
       || pgrep -f "$GHOST_PATTERN" >/dev/null; then
        warn "仍有 SteamVR 相关进程未清干净, 可手动执行:"
        warn "  pgrep -af '$GHOST_PATTERN|vrserver|vrmonitor'"
    else
        log "    幽灵树已清理干净"
    fi
}

# ---------- 2. 常驻启动完整栈 ----------
launch_steamvr() {
    log "==> 启动 SteamVR 完整栈 (steam steam://rungameid/${STEAMVR_APPID}) ..."
    if ! command -v steam >/dev/null 2>&1; then
        warn "找不到 steam 命令, 请确认 Steam 客户端已安装并在 PATH 中"
        exit 1
    fi
    steam "steam://rungameid/${STEAMVR_APPID}" >/dev/null 2>&1 &
}

# ---------- 3. 轮询等待 vrserver + vrmonitor 同时就绪 ----------
wait_ready() {
    log "==> 等待 vrserver + vrmonitor 就绪 (最长 ${WAIT_TIMEOUT}s) ..."
    local waited=0
    while [ "$waited" -lt "$WAIT_TIMEOUT" ]; do
        if pgrep -x vrserver >/dev/null && pgrep -x vrmonitor >/dev/null; then
            local vs vm
            vs="$(pgrep -x vrserver | head -n1)"
            vm="$(pgrep -x vrmonitor | head -n1)"
            log ""
            log "✅ SteamVR 常驻栈已就绪 (用时 ${waited}s)"
            log "   vrserver  PID=${vs}"
            log "   vrmonitor PID=${vm}"
            # 房间设置弹窗会挡住追踪就绪, 提示用户跳过/关闭
            if pgrep -f steamvr_room_setup >/dev/null; then
                warn "检测到 steamvr_room_setup 房间设置弹窗, 请跳过/关闭它, 否则追踪不会就绪"
            fi
            log ""
            log "现在可反复运行 client (脚本退出不影响常驻栈):"
            log "  python hardware/vive_tracker.py         # 自检"
            log "  python scripts/collect_data.py ...      # 采集"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        printf '\r    等待中... %ds/%ds (vrserver:%s vrmonitor:%s)' \
            "$waited" "$WAIT_TIMEOUT" \
            "$(pgrep -x vrserver >/dev/null && echo up || echo -)" \
            "$(pgrep -x vrmonitor >/dev/null && echo up || echo -)"
    done
    printf '\n'
    warn "等待超时 (${WAIT_TIMEOUT}s), 栈未完全就绪。排查:"
    warn "  · 无头显需配好 default.vrsettings (requireHmd:false / forcedDriver:null / driver_null.enable:true)"
    warn "  · 基站是否通电、Tracker 是否开机配对"
    warn "  · 手动查日志: ~/.local/share/Steam/logs/vrserver.txt"
    return 1
}

# ---------- 主流程 ----------
main() {
    kill_ghosts
    if [ "${1:-}" = "--kill" ]; then
        log "==> --kill 模式: 只清理, 不启动"
        return 0
    fi
    launch_steamvr
    wait_ready
}

main "$@"
