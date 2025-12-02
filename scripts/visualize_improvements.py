"""
Visual demonstration of the improved racing line algorithm

Run this to see how corner detection and line positioning works
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

def visualize_racing_line_theory():
    """Create visual explanation of racing line improvements"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Racing Line Detection Improvements', fontsize=16, fontweight='bold')
    
    # ============================================================
    # Plot 1: Track Fragmentation Problem (BEFORE)
    # ============================================================
    ax1 = axes[0, 0]
    ax1.set_title('BEFORE: Track Fragmentation Issue', fontweight='bold', color='red')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    
    # Draw fragmented track patches
    fragments = [
        Rectangle((10, 60), 40, 30, color='lightgreen', alpha=0.5, label='Main track'),
        Rectangle((60, 40), 15, 10, color='lightgreen', alpha=0.5),  # Fragment 1
        Rectangle((80, 30), 8, 8, color='lightgreen', alpha=0.5),    # Fragment 2
        Rectangle((5, 40), 10, 10, color='lightgreen', alpha=0.5),   # Fragment 3
    ]
    
    for frag in fragments:
        ax1.add_patch(frag)
    
    # Bad racing line following fragments
    bad_line_x = [20, 25, 30, 45, 67, 85]
    bad_line_y = [75, 72, 68, 60, 45, 35]
    ax1.plot(bad_line_x, bad_line_y, 'r-', linewidth=3, marker='o', 
             markersize=6, label='Bad racing line')
    
    # Car position
    ax1.plot(20, 75, 'bs', markersize=15, label='Car')
    
    ax1.text(50, 10, '❌ Line follows small fragments\n❌ Unrealistic path\n❌ Ignores main track',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 2: Track Fragmentation Fixed (AFTER)
    # ============================================================
    ax2 = axes[0, 1]
    ax2.set_title('AFTER: Connected Component Filtering', fontweight='bold', color='green')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_aspect('equal')
    
    # Only main track (fragments removed)
    main_track = Rectangle((10, 60), 70, 30, color='lightgreen', alpha=0.7, 
                          label='Main track (cleaned)')
    ax2.add_patch(main_track)
    
    # Good racing line on continuous track
    good_line_x = [20, 30, 40, 50, 60, 70]
    good_line_y = [75, 74, 73, 72, 71, 70]
    ax2.plot(good_line_x, good_line_y, 'g-', linewidth=3, marker='o', 
             markersize=6, label='Smooth racing line')
    
    # Car position
    ax2.plot(20, 75, 'bs', markersize=15, label='Car')
    
    # Show removed fragments (greyed out)
    removed = [
        Rectangle((60, 40), 15, 10, color='gray', alpha=0.2),
        Rectangle((80, 30), 8, 8, color='gray', alpha=0.2),
        Rectangle((5, 40), 10, 10, color='gray', alpha=0.2),
    ]
    for rem in removed:
        ax2.add_patch(rem)
    
    ax2.text(50, 10, '✅ Largest component selected\n✅ Small fragments ignored\n✅ Continuous track surface',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 3: Wrong Racing Line - LEFT Turn (BEFORE)
    # ============================================================
    ax3 = axes[1, 0]
    ax3.set_title('BEFORE: Wrong Line Position (Left Turn)', fontweight='bold', color='red')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.set_aspect('equal')
    
    # Draw left-turning track
    left_edge = np.array([
        [20, 90], [20, 80], [20, 70], [18, 60], [15, 50], 
        [12, 40], [10, 30], [10, 20], [12, 10]
    ])
    right_edge = np.array([
        [60, 90], [60, 80], [60, 70], [58, 60], [52, 50],
        [45, 40], [40, 30], [38, 20], [37, 10]
    ])
    
    # Fill track
    track_poly = np.vstack([left_edge, right_edge[::-1]])
    ax3.fill(track_poly[:, 0], track_poly[:, 1], color='lightgreen', 
            alpha=0.5, label='Track')
    
    # WRONG: Centerline (simple average)
    wrong_line = (left_edge + right_edge) / 2
    ax3.plot(wrong_line[:, 0], wrong_line[:, 1], 'r-', linewidth=3, 
            marker='o', markersize=5, label='Wrong: Centerline')
    
    # Car starting position (on centerline)
    ax3.plot(40, 90, 'bs', markersize=15, label='Car')
    
    # Annotations
    ax3.annotate('', xy=(35, 50), xytext=(60, 50),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax3.text(75, 50, 'Should be\nOUTSIDE\n(RIGHT)', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='white'))
    
    ax3.text(50, 5, '❌ Line on inside\n❌ Tight corner radius\n❌ Slow speed',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 4: Correct Racing Line - LEFT Turn (AFTER)
    # ============================================================
    ax4 = axes[1, 1]
    ax4.set_title('AFTER: Correct Line Position (Left Turn)', fontweight='bold', color='green')
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.set_aspect('equal')
    
    # Same track
    ax4.fill(track_poly[:, 0], track_poly[:, 1], color='lightgreen', 
            alpha=0.5, label='Track')
    
    # CORRECT: Biased line (outside before turn, late apex, wide exit)
    correct_line = np.array([
        [55, 90],   # Start RIGHT (outside)
        [56, 80],   # Stay RIGHT
        [57, 70],   # Still RIGHT
        [53, 60],   # Begin turn in
        [45, 50],   # Moving to center
        [32, 40],   # Late apex (inside)
        [28, 30],   # Past apex
        [30, 20],   # Exit wide
        [33, 10],   # Full width on exit
    ])
    ax4.plot(correct_line[:, 0], correct_line[:, 1], 'g-', linewidth=3,
            marker='o', markersize=5, label='Correct: Racing line')
    
    # Car starting position (on outside)
    ax4.plot(55, 90, 'bs', markersize=15, label='Car')
    
    # Mark key points
    ax4.plot(57, 70, 'ro', markersize=10, label='Turn-in point')
    ax4.plot(32, 40, 'yo', markersize=10, label='Late apex')
    ax4.plot(33, 10, 'co', markersize=10, label='Track-out')
    
    # Annotations
    ax4.annotate('START: Outside\n(RIGHT side)', xy=(55, 90), xytext=(70, 95),
                fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.5))
    ax4.annotate('APEX: Inside\n(Late!)', xy=(32, 40), xytext=(5, 45),
                fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.5))
    ax4.annotate('EXIT: Outside\n(Full width)', xy=(33, 10), xytext=(50, 3),
                fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.5))
    
    ax4.text(50, 95, '✅ Outside→Inside→Outside\n✅ Maximum corner radius\n✅ Optimal speed',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('racing_line_improvements.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to: racing_line_improvements.png")
    plt.show()


def visualize_corner_detection_algorithm():
    """Show how edge analysis detects corner direction"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Corner Detection Algorithm', fontsize=16, fontweight='bold')
    
    # ============================================================
    # Left Turn Detection
    # ============================================================
    ax1 = axes[0]
    ax1.set_title('LEFT Turn Detection', fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    
    # Track edges for left turn
    y_vals = np.linspace(90, 10, 20)
    left_edge_left = 30 - 0.3 * (90 - y_vals)  # Slight left movement
    right_edge_left = 70 - 0.7 * (90 - y_vals)  # Strong left movement
    
    ax1.plot(left_edge_left, y_vals, 'b-', linewidth=2, label='Left edge')
    ax1.plot(right_edge_left, y_vals, 'r-', linewidth=2, label='Right edge')
    
    # Show edge movement
    ax1.arrow(30, 90, -6, -20, head_width=2, head_length=3, fc='blue', alpha=0.5)
    ax1.arrow(70, 90, -14, -20, head_width=2, head_length=3, fc='red', alpha=0.5)
    
    ax1.text(25, 65, 'Small\nleft\nshift', fontsize=10, color='blue')
    ax1.text(50, 65, 'Large\nleft\nshift', fontsize=10, color='red', fontweight='bold')
    
    # Calculation
    left_trend = -6
    right_trend = -14
    corner_indicator = right_trend - left_trend
    
    ax1.text(50, 30, f'left_trend = {left_trend}\nright_trend = {right_trend}\n' +
            f'corner_indicator = {corner_indicator}\n\n' +
            f'→ RIGHT edge moves MORE\n→ LEFT TURN detected\n→ bias = +0.4 (RIGHT)',
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Racing line with bias
    racing_line_left = (left_edge_left + right_edge_left) / 2 + 8  # +bias to right
    ax1.plot(racing_line_left, y_vals, 'g-', linewidth=3, label='Racing line (biased RIGHT)')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ============================================================
    # Right Turn Detection
    # ============================================================
    ax2 = axes[1]
    ax2.set_title('RIGHT Turn Detection', fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_aspect('equal')
    
    # Track edges for right turn
    left_edge_right = 30 + 0.7 * (90 - y_vals)  # Strong right movement
    right_edge_right = 70 + 0.3 * (90 - y_vals)  # Slight right movement
    
    ax2.plot(left_edge_right, y_vals, 'b-', linewidth=2, label='Left edge')
    ax2.plot(right_edge_right, y_vals, 'r-', linewidth=2, label='Right edge')
    
    # Show edge movement
    ax2.arrow(30, 90, 14, -20, head_width=2, head_length=3, fc='blue', alpha=0.5)
    ax2.arrow(70, 90, 6, -20, head_width=2, head_length=3, fc='red', alpha=0.5)
    
    ax2.text(50, 65, 'Large\nright\nshift', fontsize=10, color='blue', fontweight='bold')
    ax2.text(75, 65, 'Small\nright\nshift', fontsize=10, color='red')
    
    # Calculation
    left_trend = +14
    right_trend = +6
    corner_indicator = right_trend - left_trend
    
    ax2.text(50, 30, f'left_trend = {left_trend}\nright_trend = {right_trend}\n' +
            f'corner_indicator = {corner_indicator}\n\n' +
            f'→ LEFT edge moves MORE\n→ RIGHT TURN detected\n→ bias = -0.4 (LEFT)',
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Racing line with bias
    racing_line_right = (left_edge_right + right_edge_right) / 2 - 8  # -bias to left
    ax2.plot(racing_line_right, y_vals, 'g-', linewidth=3, label='Racing line (biased LEFT)')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('corner_detection_algorithm.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to: corner_detection_algorithm.png")
    plt.show()


if __name__ == '__main__':
    print("Generating racing line improvement visualizations...\n")
    
    try:
        visualize_racing_line_theory()
        print()
        visualize_corner_detection_algorithm()
        print("\n✅ All visualizations generated successfully!")
        print("\nFiles created:")
        print("  - racing_line_improvements.png")
        print("  - corner_detection_algorithm.png")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure matplotlib is installed: pip install matplotlib")
