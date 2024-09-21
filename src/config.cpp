#include "TaskArgs.h"

#include <vector_types.h>
#include <QList>
#include <QSize>
#include <QString>

namespace ProjConf
{

extern constexpr QSize DEFAULT_MAINWINDOW_SIZE{1024, 768};

// <dGrid, dBlock>
extern QList<std::pair<dim3, dim3>> DIMENSION_OPTIONS{
    {dim3(10, 1080), dim3(192, 1)},  // 1920 x 1080
    {dim3(10, 1440), dim3(256, 1)},  // 2560 x 1440
    {dim3(15, 2160), dim3(256, 1)},  // 3840 x 2160
};

extern QString const DEFAULT_CENTER_X{QStringLiteral("-0.7438")};
extern QString const DEFAULT_CENTER_Y{QStringLiteral("0.148")};
extern QString const DEFAULT_HALF_UNIT{QStringLiteral("0.000002")};
extern QString const DEFAULT_ITER_LIMIT{QStringLiteral("4000")};

extern constexpr double LOG_NORM_SLIDER_SCALE = 10.0;
extern constexpr int LOG_NORM_SLIDER_RANGE = 65536;
extern constexpr int COLOR_RANGE_SIXTH = 1048576;  // pow(2, 20)

extern QString const INPUT_PANE_STYLE{QStringLiteral(R"(
    * {
        font-size: 20px;
    }

    QLineEdit {
        background: #8F8F8F;
        border: none;
    }
)")};

extern QString const CHANNEL_PANE_STYLE{QStringLiteral(R"(
    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #E0E0E0, stop:1 #8F8F8F);
    }
    QSlider[channel="0"]::handle {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF0000, stop:1 #F0F0F0);
    }
    QSlider[channel="1"]::handle {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00FF00, stop:1 #F0F0F0);
    }
    QSlider[channel="2"]::handle {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0000FF, stop:1 #F0F0F0);
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

extern char const* SHADER_TEXTURE_VIEW_VERTEX =
#include "shaders/TextureViewVertex.str"
    ;

extern char const* SHADER_TEXTURE_VIEW_FRAGMENT =
#include "shaders/TextureViewFragment.str"
    ;

extern constexpr float CHANNEL_CURVES_STROKE_PEAK = 1.5;
extern constexpr float CHANNEL_CURVES_STROKE_NORM = 1.0 / 0.0125;

extern char const* SHADER_CHANNEL_CURVES_VERTEX =
#include "shaders/ChannelCurvesVertex.str"
    ;

extern char const* SHADER_CHANNEL_CURVES_FRAGMENT =
#include "shaders/ChannelCurvesFragment.str"
    ;

}  // namespace ProjConf
