use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xs: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let ys: Vec<f32> = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];

    let image_width = 1080;
    let image_height = 720;
    let root = BitMapBackend::new("plot.png", (1080, 720)).into_drawing_area();

    // 背景を白にする
    root.fill(&WHITE)?;

    /* (3) グラフ全般の設定 */

    /* y軸の最大最小値を算出
    f32型はNaNが定義されていてys.iter().max()等が使えないので工夫が必要
    参考サイト
    https://qiita.com/lo48576/items/343ca40a03c3b86b67cb */
    let (y_min, y_max) = ys
        .iter()
        .fold((0.0 / 0.0, 0.0 / 0.0), |(m, n), v| (v.min(m), v.max(n)));

    let caption = "Sample Plot";
    let font = ("sans-serif", 20);

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, font.into_font()) // キャプションのフォントやサイズ
        .margin(10) // 上下左右全ての余白
        .x_label_area_size(16) // x軸ラベル部分の余白
        .y_label_area_size(42) // y軸ラベル部分の余白
        .build_cartesian_2d(
            // x軸とy軸の数値の範囲を指定する
            *xs.first().unwrap()..*xs.last().unwrap(), // x軸の範囲
            y_min..y_max,                              // y軸の範囲
        )?;

    /* (4) グラフの描画 */

    // x軸y軸、グリッド線などを描画
    chart.configure_mesh().draw()?;

    // 折れ線グラフの定義＆描画
    let line_series = LineSeries::new(xs.iter().zip(ys.iter()).map(|(x, y)| (*x, *y)), &RED);
    chart.draw_series(line_series)?;

    Ok(())
}
