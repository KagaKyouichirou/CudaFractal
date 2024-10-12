#include "InputPane.h"

#include "cuda/FixedPoint8U30.h"

#include <vector_types.h>
#include <QFontMetrics>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QIntValidator>
#include <QLabel>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <cstdint>

namespace AppConf
{
extern QList<std::pair<dim3, dim3>> const DIMENSION_OPTIONS;
extern QList<uint8_t> const FRAC_CAPACITY_OPTIONS;

extern QString const INPUT_PANE_STYLE;
extern QString const INPUT_PANE_SIGN_BTTN_STYLE;
extern QString const INPUT_PANE_LINE_EDIT_STYLE;

extern bool const DEFAULT_SIGN_CENTER_X;
extern QString const DEFAULT_FRAC_CENTER_X;
extern int const DEFAULT_EXPO_CENTER_X;

extern bool const DEFAULT_SIGN_CENTER_Y;
extern QString const DEFAULT_FRAC_CENTER_Y;
extern int const DEFAULT_EXPO_CENTER_Y;

extern QString const DEFAULT_FRAC_HALF_UNIT;
extern int const DEFAULT_EXPO_HALF_UNIT;

extern QString const DEFAULT_LINE_ITER_LIMIT;
}  // namespace AppConf

QComboBox* buildDimCombo()
{
    auto combo = new QComboBox(nullptr);
    for (auto p : AppConf::DIMENSION_OPTIONS) {
        auto w = p.first.x * p.second.x;
        auto h = p.first.y * p.second.y;
        combo->addItem(QString::asprintf("%d × %d", w, h));
    }
    return combo;
}

QPushButton* buildSignBttn()
{
    static QIcon const ICONS[2]{
        QIcon(QStringLiteral(":/icons/positive.svg")),
        QIcon(QStringLiteral(":/icons/negative.svg"))
    };
    auto bttn = new QPushButton(nullptr);
    bttn->setCheckable(true);
    bttn->setFlat(true);
    bttn->setStyleSheet(AppConf::INPUT_PANE_SIGN_BTTN_STYLE);
    bttn->setToolTip(QStringLiteral("Click to toggle the sign"));
    bttn->setIcon(ICONS[0]);
    QObject::connect(bttn, &QPushButton::toggled, [bttn](bool checked) { bttn->setIcon(ICONS[checked]); });
    return bttn;
}

QLineEdit* buildFracLine()
{
    using REV = QRegularExpressionValidator;
    static REV const validator(QRegularExpression(QStringLiteral("^[0-9a-fA-F]*$")), nullptr);
    auto line = new QLineEdit(nullptr);
    line->setStyleSheet(AppConf::INPUT_PANE_LINE_EDIT_STYLE);
    line->setValidator(&validator);
    QObject::connect(line, &QLineEdit::textEdited, [line](QString const& text) {
        auto pos = line->cursorPosition();
        auto upper = text.toUpper();
        line->setText(upper);
        line->setCursorPosition(pos);
        auto width = line->fontMetrics().horizontalAdvance(upper);
        line->setToolTip(width > line->width() ? upper : QStringLiteral("Input Hex-Digits"));
    });
    return line;
}

QComboBox* buildExpoCombo()
{
    auto combo = new QComboBox(nullptr);
    for (auto c : AppConf::FRAC_CAPACITY_OPTIONS) {
        combo->addItem(QString::number(c * 8 - 30));
    }
    return combo;
}

QLineEdit* buildLineLimit()
{
    auto line = new QLineEdit(nullptr);
    line->setStyleSheet(AppConf::INPUT_PANE_LINE_EDIT_STYLE);
    line->setValidator(new QIntValidator(1, 2147483647, line));
    return line;
}

InputPane::InputPane():
    QWidget(nullptr),
    comboDimOption(buildDimCombo()),
    signCenterX(buildSignBttn()),
    fracCenterX(buildFracLine()),
    expoCenterX(buildExpoCombo()),
    signCenterY(buildSignBttn()),
    fracCenterY(buildFracLine()),
    expoCenterY(buildExpoCombo()),
    fracHalfUnit(buildFracLine()),
    expoHalfUnit(buildExpoCombo()),
    lineIterLimit(buildLineLimit()),
    bttnRender(new QPushButton(QStringLiteral("RENDER"), nullptr)),
    bttnExport(new QPushButton(QStringLiteral("EXPORT"), nullptr))
{
    setStyleSheet(AppConf::INPUT_PANE_STYLE);

    setupLayout();

    connect(bttnRender, &QPushButton::clicked, this, &InputPane::render, Qt::DirectConnection);
    connect(bttnExport, &QPushButton::clicked, this, &InputPane::signalExportImage, Qt::DirectConnection);

    // initialize
    comboDimOption->setCurrentIndex(0);

    signCenterX->setChecked(AppConf::DEFAULT_SIGN_CENTER_X);
    fracCenterX->setText(AppConf::DEFAULT_FRAC_CENTER_X);
    emit fracCenterX->textEdited(fracCenterX->text());
    expoCenterX->setCurrentIndex(AppConf::DEFAULT_EXPO_CENTER_X);

    signCenterY->setChecked(AppConf::DEFAULT_SIGN_CENTER_Y);
    fracCenterY->setText(AppConf::DEFAULT_FRAC_CENTER_Y);
    emit fracCenterY->textEdited(fracCenterY->text());
    expoCenterY->setCurrentIndex(AppConf::DEFAULT_EXPO_CENTER_Y);

    fracHalfUnit->setText(AppConf::DEFAULT_FRAC_HALF_UNIT);
    emit fracHalfUnit->textEdited(fracHalfUnit->text());
    expoHalfUnit->setCurrentIndex(AppConf::DEFAULT_EXPO_HALF_UNIT);

    lineIterLimit->setText(AppConf::DEFAULT_LINE_ITER_LIMIT);
}

FixedPoint8U30 parseFixedPoint(QByteArray hex, uint8_t c, bool& ok)
{
    auto arr = QByteArray::fromHex(hex);
    if (arr.length() <= c) {
        std::array<uint8_t, 30> bytes{};
        size_t len = arr.length();
        size_t offset = c - len;
        std::memcpy(bytes.data() + offset, arr.data(), len);
        auto res = FixedPoint8U30::fromBytes(bytes);
        ok = res.reasonable();
        return res;
    } else {
        ok = false;
        return FixedPoint8U30();
    }
}

void InputPane::render()
{
    bool ok = false;
    // Center.X
    FixedPoint8U30 x;
    {
        if (fracCenterX->text().isEmpty()) {
            emit signalStatusTemp(QStringLiteral("Missing input for Center.X"));
            return;
        }
        static QString const errCenterX = QStringLiteral("Center.X should be within the range [-2.0, 1.0]");
        auto capacityCenterX = AppConf::FRAC_CAPACITY_OPTIONS[expoCenterX->currentIndex()];
        x = parseFixedPoint(fracCenterX->text().toUtf8(), capacityCenterX, ok);
        if (ok) {
            if (signCenterX->isChecked()) {
                static FixedPoint8U30 const boundTwo{
                    FixedPoint8U30::fromBytes(std::array<uint8_t, 30>{0, 0, 0, 8})
                };
                if (x.exceeds(boundTwo)) {
                    emit signalStatusTemp(errCenterX);
                    return;
                }
                x.flip();
            } else {
                static FixedPoint8U30 const boundOne{
                    FixedPoint8U30::fromBytes(std::array<uint8_t, 30>{0, 0, 0, 4})
                };
                if (x.exceeds(boundOne)) {
                    emit signalStatusTemp(errCenterX);
                    return;
                }
            }
        } else {
            emit signalStatusTemp(errCenterX);
            return;
        }
    }
    // Center.Y
    FixedPoint8U30 y;
    {
        if (fracCenterY->text().isEmpty()) {
            emit signalStatusTemp(QStringLiteral("Missing input for Center.Y"));
            return;
        }
        static QString const errCenterY = QStringLiteral("Center.Y should be within the range [-1.5, 1.5]");
        auto capacityCenterY = AppConf::FRAC_CAPACITY_OPTIONS[expoCenterY->currentIndex()];
        y = parseFixedPoint(fracCenterY->text().toUtf8(), capacityCenterY, ok);
        if (ok) {
            static FixedPoint8U30 const boundOneAndHalf{
                FixedPoint8U30::fromBytes(std::array<uint8_t, 30>{0, 0, 0, 6})
            };
            if (y.exceeds(boundOneAndHalf)) {
                emit signalStatusTemp(errCenterY);
                return;
            }
            if (signCenterY->isChecked()) {
                y.flip();
            }
        } else {
            emit signalStatusTemp(errCenterY);
            return;
        }
    }
    // Half Unit
    FixedPoint8U30 h;
    {
        if (fracHalfUnit->text().isEmpty()) {
            emit signalStatusTemp(QStringLiteral("Missing input for Half-Unit"));
            return;
        }
        static QString const errHalfUnit = QStringLiteral("Half-Unit should be within the range (0.0, 2^-8]");
        auto capacityHalfUnit = AppConf::FRAC_CAPACITY_OPTIONS[expoHalfUnit->currentIndex()];
        h = parseFixedPoint(fracHalfUnit->text().toUtf8(), capacityHalfUnit, ok);
        if (ok && h.nonzero()) {
            static FixedPoint8U30 const boundOneByth{
                FixedPoint8U30::fromBytes(std::array<uint8_t, 30>{0, 0, 0, 0, 4})
            };
            if (h.exceeds(boundOneByth)) {
                emit signalStatusTemp(errHalfUnit);
                return;
            }
        } else {
            emit signalStatusTemp(errHalfUnit);
            return;
        }
    }
    unsigned int l = lineIterLimit->text().toInt(&ok);
    if (!(ok && l <= 300000)) {
        emit signalStatusTemp(QStringLiteral("Iter-Limit should be at least 1 and not exceeding 300000"));
        return;
    }
    auto [dGrid, dBlock] = AppConf::DIMENSION_OPTIONS[comboDimOption->currentIndex()];
    emit signalAddTask(TaskArgs{dGrid, dBlock, x, y, h, l});
}

void InputPane::setupLayout()
{
    auto grid = new QGridLayout(this);
    grid->setContentsMargins(8, 16, 8, 0);
    grid->setColumnStretch(2, 1);

    grid->addWidget(new QLabel(QStringLiteral("Resolution")), 0, 0);
    grid->addWidget(comboDimOption, 0, 1, 1, 4);

    grid->addWidget(new QLabel(QStringLiteral("Center.X")), 1, 0);
    grid->addWidget(signCenterX, 1, 1, Qt::AlignCenter);
    grid->addWidget(fracCenterX, 1, 2);
    grid->addWidget(new QLabel(QStringLiteral("/ 2^")), 1, 3);
    grid->addWidget(expoCenterX, 1, 4);

    grid->addWidget(new QLabel(QStringLiteral("Center.Y")), 2, 0);
    grid->addWidget(signCenterY, 2, 1, Qt::AlignCenter);
    grid->addWidget(fracCenterY, 2, 2);
    grid->addWidget(new QLabel(QStringLiteral("/ 2^")), 2, 3);
    grid->addWidget(expoCenterY, 2, 4);

    grid->addWidget(new QLabel(QStringLiteral("Half Unit")), 3, 0);
    grid->addWidget(fracHalfUnit, 3, 2);
    grid->addWidget(new QLabel(QStringLiteral("/ 2^")), 3, 3);
    grid->addWidget(expoHalfUnit, 3, 4);

    grid->addWidget(new QLabel(QStringLiteral("Iter Limit")), 4, 0);
    grid->addWidget(lineIterLimit, 4, 1, 1, 3);

    auto buttons = new QWidget(nullptr);
    auto hbox = new QHBoxLayout(buttons);
    hbox->addWidget(bttnRender);
    hbox->addWidget(bttnExport);
    grid->addWidget(buttons, 5, 0, 1, 5);

    grid->setRowStretch(6, 1);
}
