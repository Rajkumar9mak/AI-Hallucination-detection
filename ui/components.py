"""
Reusable UI components for ClaimGuard AI.
Includes the 3D Breathing Particle Sphere Loader faithfully ported from Swift/SwiftUI.
"""

import streamlit as st  # type: ignore
import streamlit.components.v1 as components  # type: ignore


def render_breathing_loader(
    width: int = 280,
    height: int = 280,
    sphere_size: float = 95.0,
    breathing_speed: float = 1.3,
    rotation_speed: float = 0.35,
    dot_size: float = 4.5,
    primary_color: str = "#6366F1",    # Indigo
    secondary_color: str = "#F59E0B",  # Amber/Gold
):
    """
    Renders the 3D Breathing Particle Sphere Loader in an HTML5 Canvas.
    Faithful mathematical port of the Swift BreathingLoader particle cloud.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            html, body {{
                background: transparent;
                overflow: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                height: 100%;
            }}
            canvas {{
                background: transparent;
                display: block;
            }}
        </style>
    </head>
    <body>
        <canvas id="breathingCanvas" width="{width}" height="{height}"></canvas>

        <script>
            (function() {{
                const canvas = document.getElementById('breathingCanvas');
                const ctx = canvas.getContext('2d');
                const dpr = window.devicePixelRatio || 1;

                canvas.width = {width} * dpr;
                canvas.height = {height} * dpr;
                canvas.style.width = '{width}px';
                canvas.style.height = '{height}px';
                ctx.scale(dpr, dpr);

                const count = 170;
                const goldenAngle = Math.PI * (3 - Math.sqrt(5.0));
                const particles = [];

                // Generate spherical Fibonacci cloud
                for (let i = 0; i < count; i++) {{
                    const y = 1 - (i / (count - 1)) * 2;
                    const horizontalRadius = Math.sqrt(Math.max(0, 1 - y * y));
                    const theta = goldenAngle * i;

                    let h = (i + 1) * 0x7FEB352D;
                    h ^= (h >> 15);
                    h *= 0x846CA68B;
                    h ^= (h >> 16);
                    const phase = (Math.abs(h) % 10000) / 1000.0;

                    particles.push({{
                        id: i,
                        base: {{
                            x: Math.cos(theta) * horizontalRadius,
                            y: y,
                            z: Math.sin(theta) * horizontalRadius
                        }},
                        phase: phase,
                        colorIndex: i % 2
                    }});
                }}

                function rotateX(v, angle) {{
                    const c = Math.cos(angle);
                    const s = Math.sin(angle);
                    return {{ x: v.x, y: v.y * c - v.z * s, z: v.y * s + v.z * c }};
                }}

                function rotateY(v, angle) {{
                    const c = Math.cos(angle);
                    const s = Math.sin(angle);
                    return {{ x: v.x * c + v.z * s, y: v.y, z: -v.x * s + v.z * c }};
                }}

                const sphereSize = {sphere_size};
                const breathingSpeed = {breathing_speed};
                const rotationSpeed = {rotation_speed};
                const dotSize = {dot_size};
                const primaryColor = "{primary_color}";
                const secondaryColor = "{secondary_color}";
                const focalLength = 380.0;

                const startTime = performance.now();

                function render() {{
                    const elapsed = (performance.now() - startTime) / 1000.0;
                    const time = Math.max(0, elapsed);
                    const breath = (Math.sin(time * breathingSpeed) + 1) / 2;
                    const currentRadius = sphereSize * (80.0 / 140.0 + (60.0 / 140.0) * breath);
                    const turbulence = 22.0 * breath;
                    const expansionBonus = 3.5 * breath;

                    const rendered = particles.map(p => {{
                        let pos = {{
                            x: p.base.x * currentRadius + Math.sin(time * 3 + p.phase) * turbulence,
                            y: p.base.y * currentRadius + Math.cos(time * 4 + p.phase) * turbulence,
                            z: p.base.z * currentRadius + Math.sin(time * 5 + p.phase) * turbulence
                        }};

                        pos = rotateY(pos, time * rotationSpeed);
                        pos = rotateX(pos, time * 0.22);

                        const depthScale = focalLength / Math.max(focalLength + pos.z, 1);
                        const rearOpacity = 0.45 + 0.55 * ((pos.z + sphereSize) / (sphereSize * 2));

                        return {{
                            id: p.id,
                            x: pos.x * depthScale,
                            y: pos.y * depthScale,
                            z: pos.z,
                            size: (dotSize + expansionBonus) * depthScale,
                            opacity: pos.z < 0 ? Math.min(Math.max(rearOpacity, 0.15), 1) : 1,
                            colorIndex: p.colorIndex
                        }};
                    }});

                    // Painter's algorithm: sort rear to front
                    rendered.sort((a, b) => a.z - b.z);

                    ctx.clearRect(0, 0, {width}, {height});
                    const cx = {width} / 2;
                    const cy = {height} / 2;

                    for (const p of rendered) {{
                        ctx.save();
                        ctx.globalAlpha = p.opacity;
                        ctx.fillStyle = p.colorIndex === 0 ? primaryColor : secondaryColor;
                        ctx.beginPath();
                        ctx.arc(cx + p.x, cy + p.y, Math.max(0.8, p.size / 2), 0, Math.PI * 2);
                        ctx.fill();
                        ctx.restore();
                    }}

                    requestAnimationFrame(render);
                }}

                requestAnimationFrame(render);
            }})();
        </script>
    </body>
    </html>
    """
    components.html(html_code, width=width, height=height, scrolling=False)


def button_outline(
    label: str,
    key: str | None = None,
    help: str | None = None,
    on_click=None,
    args=None,
    kwargs=None,
    type: str = "secondary",
    disabled: bool = False,
    use_container_width: bool = False,
) -> bool:
    """
    Renders a button styled according to shadcn/ui <Button variant="outline">.

    Equivalent to:
        import { Button } from "@/components/ui/button"
        <Button variant="outline">{label}</Button>
    """
    return st.button(
        label,
        key=key,
        help=help,
        on_click=on_click,
        args=args,
        kwargs=kwargs,
        type=type,
        disabled=disabled,
        use_container_width=use_container_width,
    )
