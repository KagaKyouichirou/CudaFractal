#include "cuda/TaskArgs.h"

#include <vector_types.h>
#include <QList>
#include <QSize>
#include <QString>

namespace AppConf
{

extern constexpr QSize DEFAULT_MAINWINDOW_SIZE{1700, 1000};

extern QString const TAB_PANE_STYLE{QStringLiteral(R"(
    QTabBar::tab {
        font-size: 20px;
    }
)")};

// <dGrid, dBlock>
extern QList<std::pair<dim3, dim3>> const DIMENSION_OPTIONS{
    {dim3(10, 1080), dim3(192, 1)},  // 1920 x 1080
    {dim3(10, 1440), dim3(256, 1)},  // 2560 x 1440
    {dim3(15, 2160), dim3(256, 1)},  // 3840 x 2160
    {dim3(16, 32), dim3(32, 16)},    // 512 x 512
};

extern QList<uint8_t> const FRAC_CAPACITY_OPTIONS{6, 10, 14, 18, 22, 26, 30};

extern QString const INPUT_PANE_SIGN_BTTN_STYLE{QStringLiteral(R"(
    QPushButton {
        min-width: 24px;
        min-height: 24px;
        max-width: 24px;
        max-height: 24px;
        border: none;
    }
)")};

extern QString const INPUT_PANE_LINE_EDIT_STYLE{QStringLiteral(R"(
    QLineEdit {
        background: #D0D0D0;
        border: none;
    }
)")};

extern QString const INPUT_PANE_STYLE{QStringLiteral(R"(
    * {
        font-size: 20px;
    }
)")};

extern bool constexpr DEFAULT_SIGN_CENTER_X = false;
extern QString const DEFAULT_FRAC_CENTER_X{QStringLiteral("0")};
extern int constexpr DEFAULT_EXPO_CENTER_X = 0;

extern bool constexpr DEFAULT_SIGN_CENTER_Y = false;
extern QString const DEFAULT_FRAC_CENTER_Y{QStringLiteral("0")};
extern int constexpr DEFAULT_EXPO_CENTER_Y = 0;

extern QString const DEFAULT_FRAC_HALF_UNIT{QStringLiteral("200")};
extern int constexpr DEFAULT_EXPO_HALF_UNIT = 0;

extern QString const DEFAULT_LINE_ITER_LIMIT{QStringLiteral("4000")};

// extern bool constexpr DEFAULT_SIGN_CENTER_X = false;
// extern QString const DEFAULT_FRAC_CENTER_X{QStringLiteral("16C6672D5AF4159DAC53BB84FEE2061532328BFF00000")};
// extern int constexpr DEFAULT_EXPO_CENTER_X = 5;

// extern bool constexpr DEFAULT_SIGN_CENTER_Y = false;
// extern QString const DEFAULT_FRAC_CENTER_Y{QStringLiteral("294B8C6083878DF7482B5022E82F62230207B57000000")};
// extern int constexpr DEFAULT_EXPO_CENTER_Y = 5;

// extern QString const DEFAULT_FRAC_HALF_UNIT{QStringLiteral("40000")};
// extern int constexpr DEFAULT_EXPO_HALF_UNIT = 5;

// extern QString const DEFAULT_LINE_ITER_LIMIT{QStringLiteral("120000")};

extern QString const CHANNEL_PANE_STYLE{QStringLiteral(R"(
    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #E0E0E0, stop:1 #8F8F8F);
    }

    QSlider::groove:vertical {
        border: none;
        width: 8px;
        margin 1px 0px;
    }

    QSlider::handle:vertical {
        height: 10px;
        margin: 0 -6px;
    }

    QSlider::sub-page:vertical {
        background: #E0E0E0;
    }

    QSlider[channel="0"]::handle {
        background: qlineargradient(x1:0.5, y1:0, x2:0.5, y2:1, stop:0 #FF0000, stop:1 #FFD0D0);
    }
    QSlider[channel="0"]::add-page {
        background: #FFD0D0;
    }
    QSlider[channel="1"]::handle {
        background: qlineargradient(x1:0.5, y1:0, x2:0.5, y2:1, stop:0 #00FF00, stop:1 #D0FFD0);
    }
    QSlider[channel="1"]::add-page {
        background: #D0FFD0;
    }
    QSlider[channel="2"]::handle {
        background: qlineargradient(x1:0.5, y1:0, x2:0.5, y2:1, stop:0 #0000FF, stop:1 #D0D0FF);
    }
    QSlider[channel="2"]::add-page {
        background: #D0D0FF;
    }

    QPushButton {
        font-size: 20px;
    }

)")};

extern QString const TEXTURE_VIEW_STYLE{QStringLiteral(R"(
    min-width: 200px;
    min-height: 200px;
)")};

extern QString const CHANNEL_CURVES_STYLE{QStringLiteral(R"(
    min-width: 100px;
    min-height: 100px;
)")};

extern constexpr double GREY_NORM_SLIDER_SCALE = 20.0;
extern constexpr int GREY_NORM_SLIDER_RANGE_HALF = 65536;
extern constexpr int COLOR_RANGE_SIXTH = 1048576;  // pow(2, 20)

extern constexpr int COLOR_SLIDER_HALF_WIDTH = 10;

extern char const* SHADER_TEXTURE_VIEW_VERTEX =
#include "shaders/TextureViewVertex.str"
    ;

extern char const* SHADER_TEXTURE_VIEW_FRAGMENT =
#include "shaders/TextureViewFragment.str"
    ;

extern constexpr float CHANNEL_CURVES_STROKE_PEAK = 1.5;
extern constexpr float CHANNEL_CURVES_STROKE_NORM = 1.0 / 4;

extern char const* SHADER_CHANNEL_CURVES_VERTEX =
#include "shaders/ChannelCurvesVertex.str"
    ;

extern char const* SHADER_CHANNEL_CURVES_FRAGMENT =
#include "shaders/ChannelCurvesFragment.str"
    ;

extern char const* SHADER_TEXTURE_EXPORT_VERTEX = SHADER_CHANNEL_CURVES_VERTEX;
extern char const* SHADER_TEXTURE_EXPORT_FRAGMENT = SHADER_TEXTURE_VIEW_FRAGMENT;

}  // namespace AppConf
