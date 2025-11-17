#!/usr/bin/env python3
"""
Data Leakage Audit Script

This script audits the codebase for data leakage issues where theoretical
values are being used inappropriately in validation or analysis.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

class DataLeakageAuditor:
    """Audits codebase for data leakage issues."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.leakage_issues = []
        
        # Theoretical values that should not appear in analysis code
        self.theoretical_patterns = {
            'ising_3d_tc': r'4\.511',
            'ising_3d_beta': r'0\.326',
            'ising_3d_nu': r'0\.630',
            'ising_2d_tc': r'2\.269',
            'ising_2d_beta': r'0\.125',
            'ising_2d_nu': r'1\.0',
            'theoretical_tc': r'theoretical_tc',
            'theoretical_beta': r'theoretical_beta',
            'theoretical_nu': r'theoretical_nu'
        }
        
        # Suspicious patterns that indicate leakage
        self.suspicious_patterns = {
            'mock_perfect_results': r'MockVAECriticalExponentAnalyzer',
            'synthetic_with_theory': r'theoretical_.*\+.*noise',
            'circular_validation': r'theoretical_.*accuracy',
            'hardcoded_correlations': r'correlation.*=.*0\.9[0-9]',
            'perfect_r_squared': r'r_squared.*=.*0\.9[0-9]'
        }
    
    def audit_project(self) -> Dict[str, List[Dict]]:
        """Audit entire project for data leakage."""
        
        print("🔍 AUDITING PROJECT FOR DATA LEAKAGE")
        print("=" * 60)
        
        results = {
            'critical_issues': [],
            'suspicious_files': [],
            'mock_components': [],
            'synthetic_data_issues': [],
            'validation_issues': []
        }
        
        # Find Python files to audit
        python_files = list(self.project_root.rglob("*.py"))
        
        print(f"Scanning {len(python_files)} Python files...")
        
        for file_path in python_files:
            file_issues = self._audit_file(file_path)
            
            if file_issues:
                # Categorize issues
                for issue in file_issues:
                    if issue['severity'] == 'critical':
                        results['critical_issues'].append(issue)
                    elif 'mock' in issue['type'].lower():
                        results['mock_components'].append(issue)
                    elif 'synthetic' in issue['type'].lower():
                        results['synthetic_data_issues'].append(issue)
                    elif 'validation' in issue['type'].lower():
                        results['validation_issues'].append(issue)
                    else:
                        results['suspicious_files'].append(issue)
        
        self._print_audit_results(results)
        return results
    
    def _audit_file(self, file_path: Path) -> List[Dict]:
        """Audit a single file for leakage issues."""
        
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for theoretical value usage
            for pattern_name, pattern in self.theoretical_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    # Determine severity based on context
                    severity = self._assess_severity(file_path, line_content, pattern_name)
                    
                    issues.append({
                        'file': str(file_path),
                        'line': line_num,
                        'pattern': pattern_name,
                        'content': line_content,
                        'severity': severity,
                        'type': 'theoretical_value_usage'
                    })
            
            # Check for suspicious patterns
            for pattern_name, pattern in self.suspicious_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    issues.append({
                        'file': str(file_path),
                        'line': line_num,
                        'pattern': pattern_name,
                        'content': line_content,
                        'severity': 'high',
                        'type': 'suspicious_pattern'
                    })
        
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
        
        return issues
    
    def _assess_severity(self, file_path: Path, line_content: str, pattern_name: str) -> str:
        """Assess severity of theoretical value usage."""
        
        file_str = str(file_path).lower()
        line_lower = line_content.lower()
        
        # Critical: Usage in analysis or validation code
        if any(keyword in file_str for keyword in ['analysis', 'validation', 'extract', 'fit']):
            if any(keyword in line_lower for keyword in ['=', 'fit', 'extract', 'measure']):
                return 'critical'
        
        # Critical: Mock components using theoretical values
        if 'mock' in file_str and any(keyword in line_lower for keyword in ['exponent', 'result']):
            return 'critical'
        
        # High: Synthetic data generation using theoretical values
        if any(keyword in file_str for keyword in ['synthetic', 'generate', 'create']):
            if any(keyword in line_lower for keyword in ['magnetization', 'energy', 'config']):
                return 'high'
        
        # Medium: Test files (acceptable for validation)
        if any(keyword in file_str for keyword in ['test', 'example', 'demo']):
            return 'medium'
        
        # Low: Documentation or comments
        if line_content.strip().startswith('#') or 'print' in line_lower:
            return 'low'
        
        return 'medium'
    
    def _print_audit_results(self, results: Dict[str, List[Dict]]):
        """Print audit results in organized format."""
        
        print("\n" + "🚨 CRITICAL ISSUES (MUST FIX)" + "\n" + "=" * 60)
        
        if results['critical_issues']:
            for issue in results['critical_issues']:
                print(f"❌ {issue['file']}:{issue['line']}")
                print(f"   Pattern: {issue['pattern']}")
                print(f"   Code: {issue['content']}")
                print(f"   Type: {issue['type']}")
                print()
        else:
            print("✅ No critical issues found!")
        
        print("\n" + "⚠️  MOCK COMPONENTS (REMOVE/REPLACE)" + "\n" + "=" * 60)
        
        if results['mock_components']:
            for issue in results['mock_components']:
                print(f"🔧 {issue['file']}:{issue['line']}")
                print(f"   Pattern: {issue['pattern']}")
                print(f"   Code: {issue['content']}")
                print()
        else:
            print("✅ No mock component issues found!")
        
        print("\n" + "🧪 SYNTHETIC DATA ISSUES" + "\n" + "=" * 60)
        
        if results['synthetic_data_issues']:
            for issue in results['synthetic_data_issues']:
                print(f"🔬 {issue['file']}:{issue['line']}")
                print(f"   Pattern: {issue['pattern']}")
                print(f"   Code: {issue['content']}")
                print()
        else:
            print("✅ No synthetic data issues found!")
        
        print("\n" + "📊 VALIDATION ISSUES" + "\n" + "=" * 60)
        
        if results['validation_issues']:
            for issue in results['validation_issues']:
                print(f"📈 {issue['file']}:{issue['line']}")
                print(f"   Pattern: {issue['pattern']}")
                print(f"   Code: {issue['content']}")
                print()
        else:
            print("✅ No validation issues found!")
        
        # Summary
        total_critical = len(results['critical_issues'])
        total_issues = sum(len(issues) for issues in results.values())
        
        print("\n" + "📋 AUDIT SUMMARY" + "\n" + "=" * 60)
        print(f"Total Issues Found: {total_issues}")
        print(f"Critical Issues: {total_critical}")
        print(f"Mock Component Issues: {len(results['mock_components'])}")
        print(f"Synthetic Data Issues: {len(results['synthetic_data_issues'])}")
        print(f"Validation Issues: {len(results['validation_issues'])}")
        print(f"Other Suspicious Files: {len(results['suspicious_files'])}")
        
        if total_critical > 0:
            print(f"\n🚨 ACTION REQUIRED: {total_critical} critical issues must be fixed!")
            print("These issues cause data leakage and invalidate accuracy results.")
        else:
            print(f"\n✅ No critical data leakage detected!")
    
    def generate_fix_recommendations(self, results: Dict[str, List[Dict]]) -> List[str]:
        """Generate specific recommendations for fixing issues."""
        
        recommendations = []
        
        if results['critical_issues']:
            recommendations.append("1. IMMEDIATE: Remove all mock components that use theoretical values")
            recommendations.append("2. IMMEDIATE: Replace synthetic data generation with real Monte Carlo")
            recommendations.append("3. IMMEDIATE: Implement blind validation (no theoretical values in analysis)")
        
        if results['mock_components']:
            recommendations.append("4. Replace mock analyzers with real VAE training")
            recommendations.append("5. Remove hardcoded correlations and perfect results")
        
        if results['synthetic_data_issues']:
            recommendations.append("6. Generate data using only physical parameters (J, T, lattice size)")
            recommendations.append("7. Remove theoretical exponent usage in data generation")
        
        if results['validation_issues']:
            recommendations.append("8. Implement proper train/validation/test splits")
            recommendations.append("9. Add statistical significance testing")
            recommendations.append("10. Create cross-validation framework")
        
        return recommendations


def main():
    """Run data leakage audit."""
    
    print("🔍 DATA LEAKAGE AUDIT")
    print("=" * 60)
    print("Scanning for data leakage issues that invalidate accuracy results...")
    print()
    
    auditor = DataLeakageAuditor()
    results = auditor.audit_project()
    
    # Generate recommendations
    recommendations = auditor.generate_fix_recommendations(results)
    
    if recommendations:
        print("\n" + "🛠️  FIX RECOMMENDATIONS" + "\n" + "=" * 60)
        for rec in recommendations:
            print(f"  {rec}")
    
    print("\n" + "📄 NEXT STEPS" + "\n" + "=" * 60)
    print("1. Review DATA_LEAKAGE_FIX_PLAN.md for detailed implementation plan")
    print("2. Implement tasks 13.1-13.5 to remove data leakage")
    print("3. Implement tasks 14.1-14.3 for real VAE training")
    print("4. Expect realistic accuracy of 40-70% (not 98%)")
    print("5. Focus on honest, reproducible research")
    
    # Return exit code based on critical issues
    critical_count = len(results['critical_issues'])
    if critical_count > 0:
        print(f"\n❌ AUDIT FAILED: {critical_count} critical issues found")
        return 1
    else:
        print(f"\n✅ AUDIT PASSED: No critical data leakage detected")
        return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)