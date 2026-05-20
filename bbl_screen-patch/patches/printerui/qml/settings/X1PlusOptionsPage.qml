import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import UIBase 1.0
import X1PlusNative 1.0

import "../X1Plus.js" as X1Plus

import "qrc:/uibase/qml/widgets"
import ".."

Rectangle {
    color: Colors.gray_700

    property bool sdOffloadEnabled:  !!X1Plus.Settings.get("sd_offload.enabled",      true)
    property bool legacyWifiDriver:  !!X1Plus.Settings.get("boot.wifi_driver.bcmdhd", false)
    property bool wifiPendingReboot: false
    property bool hasExpander:       X1Plus.Expansion.hardware() != null

    function toggleWifiDriver() {
        var wifiDlgConfirm = function() {
            legacyWifiDriver = !legacyWifiDriver
            X1Plus.Settings.put("boot.wifi_driver.bcmdhd", legacyWifiDriver)
            dialogStack.popupDialog("TextConfirm", {
                name: "Reboot",
                text: qsTr("WiFi driver changes will take effect after X1Plus reboots. Would you like to reboot now?"),
                titles: [qsTr("No"), qsTr("Reboot Now")],
                defaultButton: TextConfirm.NO,
                onNo: () => X1PlusNative.system("sync; reboot")
            })
            wifiPendingReboot = true
        }
        var wifiDlg = legacyWifiDriver
            ? qsTr("X1Plus is currently configured to load legacy (OEM) WiFi driver. Would you like to use the default X1Plus WiFi driver instead?")
            : qsTr("X1Plus is currently configured to load the default X1Plus WiFi driver. Would you like to use the legacy (OEM) WiFi driver instead? (This is not typically necessary.)")
        dialogStack.popupDialog("TextConfirm", {
            name: "WiFi Driver",
            text: wifiDlg,
            titles: [qsTr("Cancel"), qsTr("Yes")],
            defaultButton: TextConfirm.YES,
            onNo: wifiDlgConfirm
        })
    }

    MarginPanel {
        id: infoPanel
        width: 400
        anchors.top: parent.top
        anchors.topMargin: 16
        anchors.left: parent.left
        anchors.leftMargin: 16
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 16
        radius: 15
        color: Colors.gray_600

        Text {
            id: infoTitle
            anchors.top: parent.top
            anchors.topMargin: 30
            anchors.left: parent.left
            anchors.leftMargin: 24
            font: Fonts.body_36
            color: Colors.gray_100
            text: "X1Plus"
        }

        ZLineSplitter {
            id: infoSplit
            anchors.top: infoTitle.bottom
            anchors.topMargin: 16
            alignment: Qt.AlignTop
            padding: 24
            color: Colors.gray_400
        }

        Text {
            anchors.top: infoSplit.bottom
            anchors.topMargin: 20
            anchors.left: parent.left
            anchors.leftMargin: 24
            anchors.right: parent.right
            anchors.rightMargin: 24
            wrapMode: Text.WordWrap
            font: Fonts.body_24
            color: Colors.gray_200
            text: qsTr("Configure X1Plus-specific features for your printer.")
        }
    }

    MarginPanel {
        anchors.left: infoPanel.right
        anchors.leftMargin: 16
        anchors.right: parent.right
        anchors.rightMargin: 16
        anchors.top: parent.top
        anchors.topMargin: 16
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 16
        radius: 15
        color: Colors.gray_600

        Flickable {
            anchors.fill: parent
            anchors.margins: 20
            contentHeight: grid.implicitHeight
            clip: true

            GridLayout {
                id: grid
                rowSpacing: 6
                columnSpacing: 12
                columns: 2
                width: parent.width

            /*** SD Card offload ***/

            Text {
                Layout.fillWidth: true
                font: Fonts.body_28
                color: hasExpander ? Colors.gray_100 : Colors.gray_400
                wrapMode: Text.Wrap
                text: !hasExpander
                    ? qsTr("SD Card Offloading — requires expansion hardware.")
                    : sdOffloadEnabled
                        ? qsTr("SD Card Offloading is enabled.")
                        : qsTr("SD Card Offloading is disabled.")
            }

            ZSwitchButton {
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                dynamicChecked: sdOffloadEnabled && hasExpander
                enabled: hasExpander
                onToggled: {
                    X1Plus.Settings.put("sd_offload.enabled", checked)
                    sdOffloadEnabled = checked
                }
            }

            Text {
                Layout.fillWidth: true
                Layout.columnSpan: 2
                font: Fonts.body_24
                color: hasExpander ? Colors.gray_200 : Colors.gray_400
                wrapMode: Text.Wrap
                text: qsTr("Redirects high read/write tasks to an attached USB drive to reduce SD wear.")
            }

            ZLineSplitter {
                Layout.fillWidth: true
                Layout.columnSpan: 2
                Layout.topMargin: 20
                Layout.bottomMargin: 10
                alignment: Qt.AlignTop
                padding: 24
                color: Colors.gray_300
            }

            /*** Expander Settings ***/

            RowLayout {
                Layout.fillWidth: true
                spacing: 12
                Image {
                    Layout.preferredWidth: 36
                    Layout.preferredHeight: 36
                    Layout.maximumWidth: 36
                    Layout.maximumHeight: 36
                    source: "../../icon/components/cfw.png"
                    fillMode: Image.PreserveAspectFit
                    opacity: hasExpander ? 1.0 : 0.4
                }
                Text {
                    Layout.fillWidth: true
                    font: Fonts.body_28
                    color: hasExpander ? Colors.gray_100 : Colors.gray_400
                    wrapMode: Text.Wrap
                    text: hasExpander
                        ? qsTr("Expander Settings")
                        : qsTr("Expander Settings — no expansion hardware detected.")
                }
            }

            ZButton {
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                text: qsTr("Configure")
                type: ZButtonAppearance.Secondary
                enabled: hasExpander
                onClicked: pageStack.push("Expansion.qml")
            }

            ZLineSplitter {
                Layout.fillWidth: true
                Layout.columnSpan: 2
                Layout.topMargin: 20
                Layout.bottomMargin: 10
                alignment: Qt.AlignTop
                padding: 24
                color: Colors.gray_300
            }

            /*** Polar Cloud ***/

            RowLayout {
                Layout.fillWidth: true
                spacing: 12
                Image {
                    Layout.preferredWidth: 36
                    Layout.preferredHeight: 36
                    Layout.maximumWidth: 36
                    Layout.maximumHeight: 36
                    source: "../../icon/components/cloud.svg"
                    fillMode: Image.PreserveAspectFit
                }
                Text {
                    Layout.fillWidth: true
                    font: Fonts.body_28
                    color: Colors.gray_100
                    wrapMode: Text.Wrap
                    text: qsTr("Polar Cloud Connection")
                }
            }

            ZButton {
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                text: qsTr("Configure")
                type: ZButtonAppearance.Secondary
                onClicked: pageStack.push("PolarCloudPage.qml")
            }

            ZLineSplitter {
                Layout.fillWidth: true
                Layout.columnSpan: 2
                Layout.topMargin: 20
                Layout.bottomMargin: 10
                alignment: Qt.AlignTop
                padding: 24
                color: Colors.gray_300
            }

            /*** WiFi driver ***/

            Text {
                Layout.fillWidth: true
                font: Fonts.body_28
                color: Colors.gray_100
                wrapMode: Text.Wrap
                text: qsTr("WiFi Driver") + (wifiPendingReboot ? qsTr(" — reboot needed") : "")
            }

            ZButton {
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                text: qsTr("Change")
                type: ZButtonAppearance.Secondary
                onClicked: toggleWifiDriver()
            }
            } // GridLayout
        } // Flickable
    }
}
