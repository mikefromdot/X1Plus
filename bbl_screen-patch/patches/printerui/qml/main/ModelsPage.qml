import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Shapes 1.12
import UIBase 1.0
import Printer 1.0
import X1PlusNative 1.0

import ".."
import "../models"
import "qrc:/uibase/qml/widgets"
import "../X1Plus.js" as X1Plus

BaseTabPage {
    property bool hadVisible: false
    onVisibleChanged: {
        DeviceManager.activeDeviceInfos(DeviceManager.DI_Storage, visible)
        if (!visible) return
        if (stack.depth === 1 && !hadVisible) {
            activePage = keys[ModelManager.perferSource]
            hadVisible = true
        }
    }
    id: tab
    subdir: "models"
    pages: [
        { name: "Preset", title: qsTr("Internal") },
        { name: "Sdcard", title: qsTr("SD Card") },
        { name: "Usb", title: qsTr("USB") },
        { name: "SdcardCache", title: qsTr("Print Cache") },
    ]
    property var keys: UIBase.enumKeys("ModelManager", "Source")
    pageHandler: function handle() {
        if (activePage === "Usb") {
            usbOverlay.visible = true;
            if (stack) stack.enabled = false;
            usbRefreshDrives();
            return true;
        }
        usbOverlay.visible = false;
        if (stack) stack.enabled = true;
        usbSelectedPath = "";
        usbSelectedMeta = null;
        usbSelectedTrays = [];
        usbPickerOpen = false;
        if (stack.depth == 1)
            ModelManager.source = keys.indexOf(tab.activePage)
        if (stack && stack.depth === 2)
           stack.pop();
    }
    Component.onCompleted: {
        activePage = "ModelList"
        pageIndicator.currentIndex = -1
    }

    // ── USB state ────────────────────────────────────────────────────────────
    property var usbDrives: []
    property string usbCurrentPath: ""
    property string usbRootPath: ""
    property var usbEntries: []
    property var usbMetaCache: ({})
    property string usbMountsRaw: ""
    property string usbSelectedPath: ""
    property var usbSelectedMeta: null
    property var usbSelectedTrays: []
    property int usbPickerFilamentIdx: 0
    property bool usbPickerOpen: false
    property bool usbUseAms: PrintManager.feeder.hasAms
    property bool usbBedLeveling: true
    property bool usbFlowCali: true
    property bool usbTimelapse: false
    property int usbSelectedPlate: 1
    property var usbCurrentPlateData: {
        if (!usbSelectedMeta) return null;
        var plates = usbSelectedMeta.plates;
        if (!plates || plates.length === 0) return usbSelectedMeta;
        var idx = usbSelectedPlate - 1;
        return (idx >= 0 && idx < plates.length) ? plates[idx] : usbSelectedMeta;
    }
    onUsbSelectedPlateChanged: {
        if (!usbSelectedMeta) return;
        var plates = usbSelectedMeta.plates;
        var plate = (plates && usbSelectedPlate >= 1 && usbSelectedPlate <= plates.length)
                    ? plates[usbSelectedPlate - 1] : usbSelectedMeta;
        usbSelectedTrays = usbAutoAssignTrays(plate);
    }

    function usbTrayLabel(idx) {
        return ["A","B","C","D"][Math.floor(idx / 4)] + (idx % 4 + 1);
    }

    function usbRefreshDrives() {
        var result = X1PlusNative.popen("awk '$2 ~ /^\\/media\\/usb[0-9]/ {dev=$1; gsub(/[0-9]+$/, \"\", dev); gsub(/.*\\//, \"\", dev); if (system(\"test -d /sys/block/\" dev) == 0) print $0}' /proc/mounts");
        var raw = result ? result.trim() : "";
        var mountsChanged = (raw !== usbMountsRaw);
        usbMountsRaw = raw;
        if (!raw) {
            usbDrives = [];
        } else {
            var seen = {};
            usbDrives = raw.split("\n").reduce(function(acc, line) {
                var parts = line.split(" ");
                var device = parts[0], mount = parts[1];
                if (mount && mount.trim().length > 0 && !seen[device]) {
                    seen[device] = true;
                    acc.push(mount);
                }
                return acc;
            }, []);
        }
        if (usbDrives.length === 0) {
            usbCurrentPath = "";
            usbRootPath = "";
            usbEntries = [];
            usbMetaCache = {};
        } else if (usbRootPath === "" || usbDrives.indexOf(usbRootPath) < 0) {
            usbMetaCache = {};
            usbRootPath = usbDrives[0];
            usbNavigateTo(usbDrives[0]);
        } else if (mountsChanged) {
            usbMetaCache = {};
            usbNavigateTo(usbCurrentPath);
        }
    }

    function usbNavigateTo(path) {
        usbCurrentPath = path;
        usbSelectedPath = "";
        usbSelectedMeta = null;
        try {
            usbEntries = JSON.parse(X1PlusNative.listDir(path)).filter(function(e) {
                return e.name.charAt(0) !== '.' && e.name !== "System Volume Information";
            });
        } catch(e) {
            usbEntries = [];
        }
    }

    function usbGetMeta(fullPath) {
        if (usbMetaCache[fullPath] !== undefined)
            return usbMetaCache[fullPath];
        var meta = { thumbnail: "", timeEstimate: 0, weightEstimate: 0.0 };
        try {
            meta = JSON.parse(X1PlusNative.parseGcodeMetadata(fullPath));
        } catch(e) {}
        usbMetaCache[fullPath] = meta;
        return meta;
    }

    function usbParentPath(path) {
        var idx = path.lastIndexOf("/");
        return idx > 0 ? path.substring(0, idx) : path;
    }

    function usbBaseName(path) {
        var idx = path.lastIndexOf("/");
        return idx >= 0 ? path.substring(idx + 1) : path;
    }

    function usbFindMatchingTray(filament) {
        if (!filament) return null;
        var targetType = (filament.type || "").toLowerCase();
        if (!targetType || targetType === "?") return null;
        var trays = PrintManager.feeder.amsTrays;
        var typeOnlyMatch = null;
        for (var i = 0; i < trays.length; i++) {
            var td = trays[i];
            if (!td.exist) continue;
            var trayType = (td.typeName + "").toLowerCase();
            var typeMatches = trayType === targetType ||
                              trayType.indexOf(targetType) >= 0 ||
                              targetType.indexOf(trayType) >= 0;
            if (!typeMatches) continue;
            if (typeOnlyMatch === null) typeOnlyMatch = td;
            if (filament.color && td.colored && Qt.colorEqual(filament.color, td.color))
                return td;
        }
        return typeOnlyMatch;
    }

    function usbTypeMatches(trayTypeName, filamentType) {
        if (!filamentType || filamentType === "?") return true;
        var tray = (trayTypeName + "").toLowerCase();
        var req  = filamentType.toLowerCase();
        return tray === req || tray.indexOf(req) >= 0 || req.indexOf(tray) >= 0;
    }

    function usbAutoAssignTrays(meta) {
        if (!meta || !meta.filaments) return [];
        var result = [];
        for (var i = 0; i < meta.filaments.length; i++)
            result.push(usbFindMatchingTray(meta.filaments[i]));
        return result;
    }

    Timer {
        id: usbDriveTimer
        interval: 5000
        running: usbOverlay.visible
        repeat: true
        onTriggered: usbRefreshDrives()
    }

    // ── USB overlay ──────────────────────────────────────────────────────────
    Item {
        id: usbOverlay
        visible: false
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.topMargin: barHeight
        anchors.bottom: parent.bottom
        z: 10

        Rectangle {
            anchors.fill: parent
            color: Colors.gray_700
        }

        Item {
            id: usbContent
            anchors.fill: parent
            anchors.margins: 16
            visible: usbSelectedPath === ""

            // No drives message
            Text {
                anchors.centerIn: parent
                visible: usbDrives.length === 0
                font: Fonts.body_38
                color: Colors.gray_500
                text: qsTr("No USB drives connected")
            }

            // Drive selector row (shown when >1 drive)
            Row {
                id: driveSelectorRow
                visible: usbDrives.length > 1
                anchors.top: parent.top
                anchors.left: parent.left
                height: usbDrives.length > 1 ? 36 : 0
                spacing: 6

                Repeater {
                    model: usbDrives
                    delegate: ZButton {
                        width: 100
                        height: 36
                        verticalTapMargin: 4
                        text: qsTr("USB %1").arg(index + 1)
                        checked: usbRootPath === modelData
                        onClicked: {
                            usbRootPath = modelData;
                            usbNavigateTo(modelData);
                        }
                    }
                }
            }

            // Current path breadcrumb (only shown when inside a subdirectory)
            Text {
                id: pathText
                visible: usbDrives.length > 0 && usbCurrentPath !== usbRootPath
                anchors.top: driveSelectorRow.bottom
                anchors.topMargin: 4
                anchors.left: parent.left
                anchors.right: parent.right
                height: visible ? 28 : 0
                verticalAlignment: Text.AlignVCenter
                font: Fonts.body_24
                color: Colors.gray_400
                elide: Text.ElideLeft
                text: usbCurrentPath.substring(usbRootPath.length)
            }

            // No files message (overlaid on grid area)
            Text {
                anchors.centerIn: usbGrid
                visible: usbDrives.length > 0 && usbGrid.model.length === 0
                font: Fonts.body_38
                color: Colors.gray_500
                text: qsTr("No files found")
                z: 1
            }

            // File grid
            GridView {
                id: usbGrid
                visible: usbDrives.length > 0
                anchors.top: pathText.bottom
                anchors.topMargin: 4
                anchors.bottom: usbContent.bottom
                width: cellWidth * 4
                anchors.horizontalCenter: parent.horizontalCenter
                cellWidth: 251 + 15
                cellHeight: 266 + 14
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                snapMode: GridView.SnapToRow
                highlightFollowsCurrentItem: false

                model: usbCurrentPath !== usbRootPath
                    ? [{name: "..", isDir: true, size: 0, isParent: true}].concat(usbEntries)
                    : usbEntries

                delegate: Item {
                    width: 252
                    height: 266

                    property string entryFullPath: usbCurrentPath + "/" + modelData.name
                    property var meta: (!modelData.isDir && !modelData.isParent)
                        ? usbGetMeta(entryFullPath)
                        : null
                    property bool isGcode: !modelData.isDir && !modelData.isParent &&
                        (modelData.name.slice(-6).toLowerCase() === ".gcode" ||
                         (modelData.name.slice(-4).toLowerCase() === ".3mf" &&
                          meta !== null && meta.hasPrintableGcode === true))
                    property bool isSelectable: !modelData.isDir && !modelData.isParent &&
                        (modelData.name.slice(-6).toLowerCase() === ".gcode" ||
                         modelData.name.slice(-4).toLowerCase() === ".3mf")

                    Rectangle {
                        id: imgPanel
                        width: 252
                        height: 188
                        radius: 15
                        color: Colors.gray_500
                        clip: true

                        Image {
                            anchors.fill: parent
                            visible: meta !== null && meta.thumbnail && meta.thumbnail.length > 0
                            source: (meta !== null && meta.thumbnail && meta.thumbnail.length > 0)
                                ? "data:image/png;base64," + meta.thumbnail : ""
                            fillMode: Image.PreserveAspectCrop
                            cache: false
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: modelData.isDir
                            font.pixelSize: 72
                            color: Colors.gray_300
                            text: modelData.isParent ? "↑" : "▶"
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: !modelData.isDir && !modelData.isParent &&
                                     (meta === null || !meta.thumbnail || meta.thumbnail.length === 0)
                            font.pixelSize: 48
                            color: Colors.gray_400
                            text: "≡"
                        }
                    }

                    MarginPanel {
                        width: parent.width
                        anchors.top: imgPanel.bottom
                        anchors.bottom: parent.bottom
                        radius: 15
                        topRadiusOff: true
                        color: Colors.gray_600

                        Text {
                            id: entryTitle
                            anchors.left: parent.left
                            anchors.leftMargin: 15
                            anchors.right: parent.right
                            anchors.rightMargin: 15
                            anchors.top: parent.top
                            anchors.topMargin: 8
                            maximumLineCount: 1
                            font: Fonts.body_24
                            color: Colors.gray_200
                            text: modelData.name
                            clip: true
                        }

                        Row {
                            id: entryInfo
                            anchors.left: parent.left
                            anchors.leftMargin: 15
                            anchors.top: entryTitle.bottom
                            anchors.topMargin: 6
                            height: 22
                            spacing: 10
                            visible: meta !== null && (meta.timeEstimate > 0 || meta.weightEstimate > 0)
                            Repeater {
                                model: {
                                    var items = [];
                                    if (meta && meta.timeEstimate > 0)
                                        items.push({ key: "time",   value: Printer.durationString(meta.timeEstimate) });
                                    if (meta && meta.weightEstimate > 0)
                                        items.push({ key: "weight", value: meta.weightEstimate.toFixed(1) + "g" });
                                    return items;
                                }
                                delegate: Row {
                                    spacing: 4
                                    Image {
                                        width: 16; height: 16
                                        anchors.verticalCenter: parent.verticalCenter
                                        source: "../../icon/" + modelData.key + ".svg"
                                        cache: false
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        font: Fonts.body_20
                                        color: Colors.gray_200
                                        text: modelData.value
                                    }
                                }
                            }
                        }
                    }

                    TapHandler {
                        onTapped: {
                            if (modelData.isParent) {
                                usbNavigateTo(usbParentPath(usbCurrentPath));
                            } else if (modelData.isDir) {
                                usbNavigateTo(usbCurrentPath + "/" + modelData.name);
                            } else if (isSelectable) {
                                var path = usbCurrentPath + "/" + modelData.name;
                                usbSelectedPath = path;
                                usbSelectedMeta = usbGetMeta(path);
                                usbSelectedTrays = usbAutoAssignTrays(usbSelectedMeta);
                                usbSelectedPlate = 1;
                                usbPickerOpen = false;
                            }
                        }
                    }
                }
            }

            SimplePager {
                anchors.right: usbGrid.right
                anchors.bottom: usbContent.bottom
                anchors.bottomMargin: 60
                visible: usbDrives.length > 0
                target: usbGrid
                pageSize: usbGrid.cellHeight * 2
                onStepTo: usbGrid.contentY = position
            }
        }

        // ── USB print confirmation ───────────────────────────────────────────
        Item {
            id: usbPrintConfirm
            anchors.fill: parent
            visible: usbSelectedPath !== ""

            MarginPanel {
                id: confirmPoster
                width: 680
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                leftMargin: 24
                topMargin: 20
                bottomMargin: 20

                color: Colors.gray_600

                Image {
                    anchors.fill: parent
                    visible: usbCurrentPlateData !== null && usbCurrentPlateData.thumbnail && usbCurrentPlateData.thumbnail.length > 0
                    source: (usbCurrentPlateData !== null && usbCurrentPlateData.thumbnail && usbCurrentPlateData.thumbnail.length > 0)
                        ? "data:image/png;base64," + usbCurrentPlateData.thumbnail : ""
                    fillMode: Image.PreserveAspectFit
                    cache: false
                }

                Text {
                    anchors.centerIn: parent
                    visible: usbCurrentPlateData === null || !usbCurrentPlateData.thumbnail || usbCurrentPlateData.thumbnail.length === 0
                    font.pixelSize: 100
                    color: Colors.gray_400
                    text: "≡"
                }

                // ── plate counter (top-right) ────────────────────────────
                Text {
                    anchors.top: parent.top
                    anchors.right: parent.right
                    anchors.topMargin: 10
                    anchors.rightMargin: 10
                    color: Colors.gray_500
                    font: Fonts.body_24
                    visible: usbSelectedMeta !== null && (usbSelectedMeta.plateCount || 1) > 1
                    text: usbSelectedPlate + " / " + (usbSelectedMeta ? (usbSelectedMeta.plateCount || 1) : 1)
                }

                // ── prev/next plate buttons overlaid on poster ───────────
                ZButton {
                    visible: usbSelectedMeta !== null && (usbSelectedMeta.plateCount || 1) > 1
                    x: 32
                    height: width
                    anchors.verticalCenter: parent.verticalCenter
                    type: ZButtonAppearance.Secondary
                    iconSize: 0
                    cornerRadius: width / 2
                    rotation: -90
                    icon: "../../icon/up.svg"
                    enabled: usbSelectedPlate > 1
                    onClicked: usbSelectedPlate--
                }
                ZButton {
                    visible: usbSelectedMeta !== null && (usbSelectedMeta.plateCount || 1) > 1
                    height: width
                    anchors.right: parent.right
                    anchors.rightMargin: 32
                    anchors.verticalCenter: parent.verticalCenter
                    type: ZButtonAppearance.Secondary
                    iconSize: 0
                    cornerRadius: width / 2
                    rotation: 90
                    icon: "../../icon/up.svg"
                    enabled: usbSelectedPlate < (usbSelectedMeta ? (usbSelectedMeta.plateCount || 1) : 1)
                    onClicked: usbSelectedPlate++
                }

                Shape {
                    id: posterInfoBar
                    width: parent.width
                    height: 112
                    anchors.bottom: parent.bottom
                    property real barRadius: 15

                    ShapePath {
                        fillColor: "#66000000"
                        strokeColor: "transparent"
                        startX: 0; startY: 0
                        PathLine { x: posterInfoBar.width; y: 0 }
                        PathLine { x: posterInfoBar.width; y: posterInfoBar.height - posterInfoBar.barRadius }
                        PathArc  { radiusX: posterInfoBar.barRadius; radiusY: posterInfoBar.barRadius
                                   x: posterInfoBar.width - posterInfoBar.barRadius; y: posterInfoBar.height }
                        PathLine { x: posterInfoBar.barRadius; y: posterInfoBar.height }
                        PathArc  { radiusX: posterInfoBar.barRadius; radiusY: posterInfoBar.barRadius
                                   x: 0; y: posterInfoBar.height - posterInfoBar.barRadius }
                        PathLine { x: 0; y: 0 }
                    }

                    Text {
                        id: posterTitle
                        anchors.left: parent.left
                        anchors.leftMargin: 15
                        anchors.top: parent.top
                        anchors.topMargin: 10
                        width: parent.width - 30
                        font: Fonts.head_30
                        color: Colors.white_900
                        elide: Text.ElideRight
                        clip: true
                        text: usbBaseName(usbSelectedPath)
                    }

                    ZLineSplitter {
                        id: posterInfoSplit
                        alignment: Qt.AlignTop
                        anchors.top: posterTitle.bottom
                        anchors.topMargin: 10
                        padding: 20
                        color: Colors.brand
                    }

                    ListView {
                        height: 47
                        anchors.left: posterInfoSplit.left
                        anchors.right: posterInfoSplit.right
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 5
                        orientation: ListView.Horizontal
                        boundsBehavior: ListView.StopAtBounds
                        model: {
                            var items = [];
                            if (usbCurrentPlateData && usbCurrentPlateData.timeEstimate > 0)
                                items.push({ key: "time",   value: Printer.durationString(usbCurrentPlateData.timeEstimate) });
                            if (usbCurrentPlateData && usbCurrentPlateData.weightEstimate > 0)
                                items.push({ key: "weight", value: usbCurrentPlateData.weightEstimate.toFixed(1) + qsTr("g") });
                            return items;
                        }
                        delegate: Item {
                            width: ListView.view.width / 3
                            height: ListView.view.height
                            Image {
                                id: posterInfoIcon
                                anchors.left: parent.left
                                anchors.leftMargin: 20
                                anchors.verticalCenter: parent.verticalCenter
                                source: "../../icon/" + modelData.key + ".svg"
                                cache: false
                            }
                            Text {
                                anchors.left: posterInfoIcon.right
                                anchors.leftMargin: 5
                                anchors.verticalCenter: parent.verticalCenter
                                font: Fonts.body_30
                                color: Colors.gray_100
                                text: modelData.value
                            }
                        }
                    }
                }
            }

            MarginPanel {
                id: infoPanel
                anchors.left: confirmPoster.right
                anchors.leftMargin: 23
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                leftMargin: 23
                rightMargin: 21
                topMargin: confirmPoster.topMargin
                bottomMargin: confirmPoster.bottomMargin
                color: Colors.gray_800

                // ── not-printable message ────────────────────────────────
                Text {
                    anchors.top: parent.top
                    anchors.topMargin: 30
                    anchors.left: parent.left
                    anchors.leftMargin: 40
                    anchors.right: parent.right
                    anchors.rightMargin: 20
                    visible: usbSelectedMeta !== null && usbSelectedMeta.hasPrintableGcode !== true
                    font: Fonts.body_26
                    color: Colors.gray_400
                    horizontalAlignment: Text.AlignHCenter
                    wrapMode: Text.WordWrap
                    text: qsTr("Model only\nCannot print directly")
                }

                // ── filament header ──────────────────────────────────────
                Text {
                    id: filamentHeader
                    visible: usbSelectedMeta !== null && usbSelectedMeta.hasPrintableGcode === true
                    anchors.top: parent.top
                    anchors.topMargin: 15
                    anchors.left: parent.left
                    anchors.leftMargin: 40
                    font: Fonts.body_24
                    color: Colors.gray_300
                    text: usbPickerOpen
                        ? qsTr("Select AMS tray")
                        : (usbUseAms && PrintManager.feeder.hasAms
                            ? qsTr("Filament selection") : qsTr("Filament"))
                }

                // ── filament swatches ────────────────────────────────────
                Item {
                    id: swatchContainer
                    visible: filamentHeader.visible
                    anchors.top: filamentHeader.bottom
                    anchors.topMargin: 10
                    anchors.left: parent.left
                    anchors.leftMargin: 40
                    anchors.right: parent.right
                    anchors.rightMargin: 40
                    height: swatchFlow.height

                    property bool amsActive: usbUseAms && PrintManager.feeder.hasAms

                    Flow {
                        id: swatchFlow
                        width: parent.width
                        spacing: 8

                        Repeater {
                            model: (usbCurrentPlateData && usbCurrentPlateData.filaments)
                                   ? usbCurrentPlateData.filaments : []

                            delegate: Item {
                                id: swatchItem
                                width: 120
                                height: 110

                                property int filamentIdx: index
                                property var assignedTray: swatchItem.filamentIdx < usbSelectedTrays.length
                                                           ? usbSelectedTrays[swatchItem.filamentIdx] : null
                                property color swatchColor: modelData ? modelData.color : "#808080"
                                property color swatchTextColor: {
                                    var c = swatchItem.swatchColor;
                                    var lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
                                    return lum > 0.5 ? "#000000" : "#ffffff";
                                }
                                property string swatchType: modelData ? (modelData.type || "?") : "?"

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 10
                                    color: swatchItem.swatchColor

                                    Text {
                                        anchors.fill: parent
                                        anchors.margins: 5
                                        anchors.bottomMargin: swatchContainer.amsActive ? 22 : 5
                                        font: Fonts.body_24
                                        minimumPixelSize: 8
                                        fontSizeMode: Text.Fit
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                        color: swatchItem.swatchTextColor
                                        text: swatchItem.swatchType
                                    }

                                    Rectangle {
                                        id: trayBadge
                                        visible: swatchContainer.amsActive
                                        width: parent.width
                                        height: 22
                                        anchors.bottom: parent.bottom
                                        color: (swatchItem.assignedTray && swatchItem.assignedTray.colored)
                                               ? swatchItem.assignedTray.color : Qt.rgba(0, 0, 0, 0.55)
                                        radius: 10
                                        property color badgeTextColor: {
                                            var c = trayBadge.color;
                                            var lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
                                            return lum > 0.5 ? "#000000" : "#ffffff";
                                        }
                                        Rectangle {
                                            width: parent.width
                                            height: 10
                                            anchors.top: parent.top
                                            color: parent.color
                                        }
                                        Text {
                                            anchors.centerIn: parent
                                            font: Fonts.body_16
                                            color: trayBadge.badgeTextColor
                                            text: swatchItem.assignedTray
                                                  ? usbTrayLabel(swatchItem.assignedTray.index) : "--"
                                        }
                                    }

                                    Rectangle {
                                        anchors.fill: parent
                                        anchors.margins: -3
                                        color: "transparent"
                                        radius: 13
                                        border.width: (usbPickerOpen &&
                                                       usbPickerFilamentIdx === swatchItem.filamentIdx) ? 2 : 0
                                        border.color: Colors.brand
                                    }
                                }

                                TapHandler {
                                    gesturePolicy: TapHandler.ReleaseWithinBounds
                                    enabled: swatchContainer.amsActive
                                    onTapped: {
                                        if (usbPickerOpen && usbPickerFilamentIdx === swatchItem.filamentIdx) {
                                            usbPickerOpen = false;
                                        } else {
                                            usbPickerFilamentIdx = swatchItem.filamentIdx;
                                            usbPickerOpen = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ── AMS tray picker ───────────────────────────────────────
                Rectangle {
                    id: usbTrayPicker
                    visible: usbPickerOpen && usbUseAms && PrintManager.feeder.hasAms
                    z: 5
                    anchors.top: swatchContainer.bottom
                    anchors.topMargin: 8
                    anchors.left: parent.left
                    anchors.leftMargin: 30
                    anchors.right: parent.right
                    anchors.rightMargin: 30
                    height: Math.ceil((PrintManager.feeder.hasAms ? PrintManager.feeder.amsTrays.length : 0) / 4) * 68 + 16
                    radius: 12
                    color: Colors.gray_500

                    GridView {
                        x: 8; y: 8
                        width: parent.width - 16
                        height: parent.height - 16
                        cellWidth: Math.floor(width / 4)
                        cellHeight: 68
                        interactive: false
                        model: PrintManager.feeder.hasAms ? PrintManager.feeder.amsTrays : []

                        delegate: Item {
                            property var td: modelData
                            width: GridView.view.cellWidth - 8
                            height: GridView.view.cellHeight - 8
                            property bool compatible: {
                                if (!td.exist) return false;
                                var filaments = usbCurrentPlateData ? usbCurrentPlateData.filaments : null;
                                if (!filaments || usbPickerFilamentIdx >= filaments.length) return true;
                                return usbTypeMatches(td.typeName + "",
                                                      filaments[usbPickerFilamentIdx].type || "?");
                            }
                            opacity: compatible ? 1.0 : 0.3

                            Rectangle {
                                id: trayCell
                                anchors.fill: parent
                                radius: 10
                                color: td.exist && td.colored ? td.color : Colors.gray_600
                                border.width: 1
                                border.color: Colors.gray_400
                                property color cellTextColor: {
                                    var c = trayCell.color;
                                    var lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
                                    return lum > 0.5 ? "#000000" : "#ffffff";
                                }

                                Text {
                                    anchors.top: parent.top
                                    anchors.topMargin: 3
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    font: Fonts.body_16
                                    color: trayCell.cellTextColor
                                    text: usbTrayLabel(td.index)
                                }

                                Text {
                                    anchors.fill: parent
                                    anchors.topMargin: 20
                                    anchors.margins: 4
                                    font: Fonts.body_20
                                    minimumPixelSize: 8
                                    fontSizeMode: Text.Fit
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    color: trayCell.cellTextColor
                                    text: td.exist ? (td.typeName + "") : "—"
                                }
                            }

                            TapHandler {
                                gesturePolicy: TapHandler.ReleaseWithinBounds
                                enabled: compatible
                                onTapped: {
                                    var copy = usbSelectedTrays.slice();
                                    copy[usbPickerFilamentIdx] = td;
                                    usbSelectedTrays = copy;
                                    usbPickerOpen = false;
                                }
                            }
                        }
                    }
                }

                // ── checkboxes ───────────────────────────────────────────
                ListView {
                    id: switchList
                    anchors.bottom: infoSplit.top
                    anchors.bottomMargin: 15
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.margins: 30
                    height: contentHeight
                    visible: usbSelectedMeta !== null && usbSelectedMeta.hasPrintableGcode === true
                    spacing: 8
                    model: [
                        { text: qsTr("Use AMS"),          checked: usbUseAms,      vis: PrintManager.feeder.hasAms, key: "ams" },
                        { text: qsTr("Bed Leveling"),     checked: usbBedLeveling, vis: true,                       key: "bed" },
                        { text: qsTr("Flow Calibration"), checked: usbFlowCali,    vis: true,                       key: "flow" },
                        { text: qsTr("Timelapse"),        checked: usbTimelapse,   vis: true,                       key: "tl" }
                    ]
                    delegate: Item {
                        width: parent.width
                        height: modelData.vis ? checkBox.height : 0
                        visible: modelData.vis
                        ZCheckBox {
                            id: checkBox
                            font: Fonts.body_28
                            textColor: StateColors.get("gray_100")
                            text: modelData.text
                            tapMargin: 5
                            checked: modelData.checked
                            onCheckedChanged: {
                                if (modelData.key === "ams")  usbUseAms      = checked;
                                else if (modelData.key === "bed")  usbBedLeveling = checked;
                                else if (modelData.key === "flow") usbFlowCali    = checked;
                                else if (modelData.key === "tl")   usbTimelapse   = checked;
                            }
                        }
                    }
                }

                ZLineSplitter {
                    id: infoSplit
                    alignment: Qt.AlignBottom
                    padding: 30
                    offset: 122
                    color: Colors.brand
                }

                ZButton {
                    id: backBtn
                    width: 131
                    anchors.left: infoSplit.left
                    anchors.top: infoSplit.bottom
                    anchors.topMargin: 29
                    paddingX: 15
                    verticalTapMargin: 10
                    text: qsTr("Back")
                    onClicked: {
                        usbSelectedPath = "";
                        usbSelectedMeta = null;
                        usbSelectedTrays = [];
                        usbPickerOpen = false;
                    }
                }

                ZButton {
                    width: 200
                    anchors.right: infoSplit.right
                    anchors.top: backBtn.top
                    verticalTapMargin: 10
                    checked: true
                    visible: usbSelectedMeta !== null && usbSelectedMeta.hasPrintableGcode === true
                    text: qsTr("Print now")
                    onClicked: {
                        var path = usbSelectedPath;
                        var meta = usbSelectedMeta;
                        usbSelectedPath = "";
                        usbSelectedMeta = null;
                        var is3mf = path.slice(-4).toLowerCase() === ".3mf";
                        var amsMapping = (function() {
                            if (!usbUseAms || usbSelectedTrays.length === 0) return [];
                            var anyAssigned = false;
                            for (var i = 0; i < usbSelectedTrays.length; i++)
                                if (usbSelectedTrays[i]) { anyAssigned = true; break; }
                            if (!anyAssigned) return [];
                            return usbSelectedTrays.map(function(t) { return t ? t.index : 0; });
                        })();
                        var printPayload = is3mf ? {
                            command: "project_file",
                            param: "Metadata/plate_" + usbSelectedPlate + ".gcode",
                            url: "file://" + path,
                            project_id: "0",
                            profile_id: "0",
                            task_id: "0",
                            subtask_id: "0",
                            subtask_name: "",
                            md5: "",
                            timelapse: usbTimelapse,
                            bed_type: "auto",
                            bed_levelling: usbBedLeveling,
                            flow_cali: usbFlowCali,
                            vibration_cali: true,
                            layer_inspect: true,
                            ams_mapping: amsMapping,
                            use_ams: usbUseAms,
                            sequence_id: "0"
                        } : {
                            command: "gcode_file",
                            param: path,
                            sequence_id: 0,
                            use_ams: usbUseAms,
                            bed_leveling: usbBedLeveling,
                            flow_cali: usbFlowCali,
                            timelapse: usbTimelapse
                        };
                        X1Plus.lastUsbPrint = { payload: printPayload };
                        X1Plus.DDS.publish("device/request/print", printPayload);
                        navigator.activePage = "Home";
                    }
                }
            }
        }
    }
}
